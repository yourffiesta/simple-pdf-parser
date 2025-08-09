import os
import asyncio
import fitz
import argparse
import logging
from glob import glob
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini API 호출을 위한 기본 프롬프트 (기존 방식과 유사하게)
EXTRACT_TEXT_PROMPT = """
당신은 PDF 문서를 분석하여 JSON 형식으로 출력하는 AI입니다.

[Goal]
PDF 문서를 페이지 순서, 그리고 페이지 내 위에서 아래 순서로 스캔하여 모든 요소를 찾고, 아래 JSON Output Schema를 엄격히 준수하는 JSON 객체를 생성하세요.

[Key Rules]
1. 단어 중간의 불필요한 줄 바꿈을 제거하여 문법적으로 자연스러운 텍스트를 생성합니다.
2. header와 footer는 결과에 포함하지 않습니다.
3. 제목은 "##"를 앞에 붙여서 마크다운 형식으로 표현합니다.
4. 표(table)는 마크다운 테이블 형식으로 변환합니다.
5. 각 페이지의 마지막 paragraph가 문법적으로 끝나지 않은 경우, is_incomplete 필드를 true로 설정합니다.

[JSON Output Schema]
{
  "data": [
    {
      "type": "sub_title",  // 또는 "paragraph", "table"
      "page_index": 0,  // zero based physical page index
      "content": "추출된 텍스트 또는 마크다운 형식의 테이블",
      "is_incomplete": false  // 문장이 미완성일 경우 true
    }
  ]
}
"""


class SimplePDFExtractor:
    """PDF에서 텍스트만 추출하는 단순화된 클래스"""

    def __init__(
        self, output_dir="output", api_key=None, chunk_size=3, concurrency_limit=5
    ):
        self.output_dir = output_dir
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.chunk_size = chunk_size
        self.concurrency_limit = concurrency_limit

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not provided or set in environment.")

        self.client = genai.Client(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

    def _split_pdf(self, doc, chunk_size=None):
        """PDF를 지정된 크기의 청크로 분할"""
        if chunk_size is None:
            chunk_size = self.chunk_size

        chunks = []
        for i in range(0, doc.page_count, chunk_size):
            start_page = i
            end_page = min(i + chunk_size, doc.page_count)

            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

            pdf_chunk_bytes = chunk_doc.tobytes()
            chunks.append((pdf_chunk_bytes, start_page))
            chunk_doc.close()

        return chunks

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _call_gemini_api(self, pdf_chunk_bytes):
        """Gemini API를 호출하고 JSON 결과를 반환"""
        prompt_parts = [
            types.Part.from_text(text=EXTRACT_TEXT_PROMPT),
            types.Part.from_bytes(data=pdf_chunk_bytes, mime_type="application/pdf"),
        ]
        contents = [types.Content(role="user", parts=prompt_parts)]
        config = types.GenerateContentConfig(response_mime_type="application/json")

        logger.info("청크를 Gemini API에 전송합니다...")
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash", contents=contents, config=config
        )
        logger.info("Gemini API 응답을 수신했습니다.")

        try:
            import json

            return json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            logger.error(f"응답 텍스트: {response.text[:500]}...")
            raise

    async def _process_chunk(self, pdf_chunk_bytes, base_page_index):
        """단일 PDF 청크를 처리하고 페이지 인덱스를 재조정"""
        async with self.semaphore:
            try:
                result_json = await self._call_gemini_api(pdf_chunk_bytes)

                # 페이지 인덱스 재조정
                if "data" in result_json:
                    for item in result_json["data"]:
                        if "page_index" in item:
                            item["page_index"] += base_page_index

                return result_json
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    logger.warning(
                        f"API 할당량 초과 (페이지 {base_page_index}): {error_msg}"
                    )
                    logger.info("잠시 대기 후 재시도합니다...")
                    await asyncio.sleep(10)  # 10초 대기
                logger.error(f"청크 처리 실패(시작 페이지 {base_page_index}): {e}")
                return None

    def _merge_results(self, results):
        """여러 청크의 결과(JSON)를 하나로 병합"""
        final_json = {"data": []}
        for res in results:
            if res and "data" in res:
                final_json["data"].extend(res["data"])

        # 최종적으로 페이지 인덱스 순으로 정렬
        final_json["data"].sort(key=lambda x: x.get("page_index", 0))
        return final_json

    def _json_to_text(self, json_data):
        """JSON 데이터를 마크다운 형식의 텍스트로 변환"""
        if not json_data or "data" not in json_data or not json_data["data"]:
            return ""

        text_parts = []
        current_page = -1
        prev_is_incomplete = False
        prev_content = ""

        for i, item in enumerate(json_data["data"]):
            item_type = item.get("type", "paragraph")
            page_index = item.get("page_index", 0)
            content = item.get("content", "")
            is_incomplete = item.get("is_incomplete", False)

            # 이전 문단이 미완성이었다면 현재 내용과 연결
            if prev_is_incomplete and item_type == "paragraph":
                content = prev_content + " " + content
                # 이전 항목 제거
                if text_parts:
                    text_parts.pop()

            # 페이지가 바뀔 때마다 페이지 표시 추가
            if page_index != current_page:
                if text_parts:  # 첫 페이지가 아닌 경우 줄바꿈 추가
                    text_parts.append("")
                text_parts.append(f"[page_index: {page_index}]")
                current_page = page_index

            # 타입별로 적절한 형식으로 추가
            if item_type == "sub_title":
                text_parts.append(f"## {content}")
            elif item_type == "table":
                text_parts.append(content)  # 이미 마크다운 형식
            else:  # paragraph 또는 기타
                text_parts.append(content)

            prev_is_incomplete = is_incomplete
            prev_content = content

        return "\n\n".join(text_parts)

    async def extract_text(self, pdf_path):
        """PDF에서 텍스트를 추출하고 파일로 저장"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return

        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

        logger.info(f"PDF 열기: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"PDF 열기 실패: {pdf_path} (사유: {e})")
            return

        logger.info("PDF를 청크로 분할합니다...")
        pdf_chunks = self._split_pdf(doc)

        tasks = [
            self._process_chunk(chunk_bytes, base_index)
            for chunk_bytes, base_index in pdf_chunks
        ]

        logger.info(f"총 {len(tasks)}개 청크를 동시 처리합니다...")
        results = await asyncio.gather(*tasks)

        logger.info("청크 결과를 병합합니다...")
        final_json = self._merge_results(results)

        # JSON을 텍스트로 변환
        final_text = self._json_to_text(final_json)

        # 텍스트 파일로 저장
        output_path = os.path.join(self.output_dir, f"{base_filename}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        logger.info(f"텍스트 추출 완료: {output_path}")

        doc.close()


async def main(input_path, output_dir, api_key=None):
    """메인 함수"""
    extractor = SimplePDFExtractor(output_dir=output_dir, api_key=api_key)

    if os.path.isdir(input_path):
        logger.info(f"디렉터리 내 모든 PDF 처리: {input_path}")
        pdf_files = glob(os.path.join(input_path, "*.pdf"))
        if not pdf_files:
            logger.warning("디렉터리에서 PDF 파일을 찾지 못했습니다.")
            return

        # 동시 실행할 태스크 목록 생성
        tasks = [extractor.extract_text(pdf_file) for pdf_file in pdf_files]
        await asyncio.gather(*tasks)

    elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        await extractor.extract_text(input_path)

    else:
        logger.error(f"유효하지 않은 경로입니다: '{input_path}'")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="PDF에서 텍스트를 추출합니다.")
    parser.add_argument(
        "input_path", help="입력 PDF 파일 경로 또는 PDF 파일이 포함된 디렉터리 경로"
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="출력 텍스트 파일을 저장할 디렉터리. 기본값은 'output'",
    )
    parser.add_argument(
        "--api_key", help="Gemini API 키. 미지정 시 GEMINI_API_KEY 환경변수를 사용"
    )

    args = parser.parse_args()

    # API 키 확인
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API 키가 설정되지 않았습니다.")
        logger.error(
            "--api_key 옵션을 제공하거나 GEMINI_API_KEY 환경변수를 설정하세요."
        )
        exit(1)

    # 메인 비동기 함수 실행
    asyncio.run(main(args.input_path, args.output_dir, api_key))
