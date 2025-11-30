# HƯỚNG DẪN DUANSINHDE (Docker Compose)

## 1) Tổng quan luồng chạy
Bạn có 2 service:

### A) Backend (FastAPI) – `api.py`
- `GET /worldbank/countries`: lấy danh sách quốc gia từ World Bank
- `GET /worldbank/series/{country_id}?start_year&end_year`: lấy chuỗi dữ liệu birth/death/pop
- `POST /ai/analyze`: nhận `summary_text` và gọi Gemini (qua `gpt_utils.py`) trả về `markdown`

Gemini API key lấy từ `.env`:
- `GEMINI_API_KEY`
- `GEMINI_MODEL` (mặc định gemini-2.5-flash)

### B) Frontend (Streamlit) – `main.py`
- Gọi backend để:
  - load countries
  - fetch series theo country_id + year range
  - khi bấm “Phân tích bằng Gemini” thì POST summary_text lên `/ai/analyze`
- Render markdown bằng thư viện `markdown` (file `md_utils.py`) nên hạn chế lỗi markdown bị “dính code block”.

## 2) Các file chính
- `main.py`: UI Streamlit
- `api.py`: API FastAPI
- `data_utils.py`: WorldBank fetch & normalize dataframe
- `gpt_utils.py`: Gemini service (đọc env, call model)
- `md_utils.py`: normalize + render markdown -> HTML box scroll
- `exporters.py`: xuất PDF/PPTX
- `docker-compose.yml`: chạy 2 service FE/BE
- `Dockerfile`: image chung cho cả FE/BE

## 3) Chạy bằng Docker Compose
### Bước 1: tạo .env
- Copy `.env.example` thành `.env`
- Dán `GEMINI_API_KEY` thật

### Bước 2: build + run
```bash
docker compose up --build
