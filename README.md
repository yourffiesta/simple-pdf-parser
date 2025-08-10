# Simple PDF Parser

PDF에서 텍스트를 추출하는 도구임. Gemini API를 사용하여 PDF 내용을 분석하고 텍스트로 변환함.

## 기능

- PDF를 3페이지씩 청크로 분할하여 병렬 처리
- 텍스트 기반/이미지 기반 PDF 모두 처리 가능
- 마크다운 형식으로 제목과 표 변환
- 페이지별 인덱스 보존

## 설치

```bash
# uv 사용 시
uv sync
```

## 사용법

```bash
# 단일 PDF 처리
uv run simple_pdf_parser.py sample.pdf

# 디렉터리 내 모든 PDF 처리
uv run simple_pdf_parser.py pdfs/

# 출력 디렉터리 지정
uv run simple_pdf_parser.py sample.pdf --output_dir results/

# API 키 직접 지정
uv run simple_pdf_parser.py sample.pdf --api_key YOUR_API_KEY
```

## 환경 설정

`.env` 파일 생성 또는 환경 변수 설정:

```bash
GEMINI_API_KEY="your_gemini_api_key"
```

## 출력

- 추출된 텍스트는 `*.txt` 파일로 저장됨
- 기본 출력 디렉터리: `output/`

## 요구사항

- Python 3.13+
- Gemini API 키 필요
