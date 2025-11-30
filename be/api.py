# be/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from be.data_utils import get_country_list_worldbank, get_series_for_country
from be.gpt_utils import gemini_analyze_summary

app = FastAPI(title="Population API")

# (tùy chọn nhưng nên có) để FE gọi backend không bị CORS khi deploy/tách domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class AnalyzeReq(BaseModel):
    summary_text: str


class ReviseReq(BaseModel):
    # hỗ trợ nhiều format để tương thích FE/backwards
    report_markdown: str | None = None
    report_text: str | None = None
    markdown: str | None = None
    report: str | None = None

    edit_request: str | None = None
    request: str | None = None

    system_prompt: str | None = None


DEFAULT_REVISE_PROMPT = """
Bạn sẽ chỉnh sửa báo cáo markdown hiện có theo yêu cầu người dùng.
Yêu cầu:
- Giữ markdown gọn gàng, nhất quán.
- Không lặp lại toàn bộ nội dung không cần thiết.
- Chỉ trả về phiên bản báo cáo markdown cuối cùng (không thêm giải thích, không thêm lời dẫn).
""".strip()


# ---------- Utils ----------
def df_to_records_safe(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    clean = df.copy()
    clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def _pick_first_non_empty(*vals: str | None) -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/worldbank/countries")
def countries():
    return get_country_list_worldbank()


@app.get("/worldbank/series/{country_id}")
def series(country_id: str, start_year: int, end_year: int):
    try:
        df = get_series_for_country(country_id, start_year, end_year)
        return df_to_records_safe(df)
    except HTTPException:
        raise
    except Exception as e:
        # trả lỗi rõ ràng, không “500 mù”
        raise HTTPException(status_code=400, detail=f"Cannot load series for '{country_id}': {e}")


@app.post("/ai/analyze")
def analyze(req: AnalyzeReq):
    if not (req.summary_text or "").strip():
        raise HTTPException(status_code=400, detail="summary_text is empty")
    try:
        md = gemini_analyze_summary(req.summary_text)
        if not (md or "").strip():
            raise HTTPException(status_code=502, detail="Gemini returned empty response")
        return {"markdown": md}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# nhận cả 2 path để tránh lỗi / và không /
@app.post("/ai/revise")
@app.post("/ai/revise/")
def revise(req: ReviseReq):
    report_md = _pick_first_non_empty(req.report_markdown, req.report_text, req.markdown, req.report)
    edit_req = _pick_first_non_empty(req.edit_request, req.request)
    system_prompt = (req.system_prompt or DEFAULT_REVISE_PROMPT).strip()

    if not report_md:
        raise HTTPException(status_code=400, detail="report markdown is empty")
    if not edit_req:
        raise HTTPException(status_code=400, detail="edit_request is empty")

    # Dùng chung hàm gemini_analyze_summary: coi như “1 prompt lớn”
    prompt = "\n".join(
        [
            "### SYSTEM PROMPT",
            system_prompt,
            "",
            "### BÁO CÁO HIỆN TẠI (MARKDOWN)",
            report_md,
            "",
            "### YÊU CẦU CHỈNH SỬA",
            edit_req,
        ]
    ).strip()

    try:
        md = gemini_analyze_summary(prompt)
        if not (md or "").strip():
            raise HTTPException(status_code=502, detail="Gemini returned empty response")
        return {"markdown": md}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
