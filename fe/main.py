import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from fe.api_client import get_countries, get_series, analyze, revise
from fe.prompts import BASE_REVISE_PROMPT, build_ai_input
from fe.data_helpers import normalize_series_df, safe_mean, safe_trend
from fe.export_ai import markdown_to_plain_text, pdf_bytes_from_text, docx_bytes_from_text, safe_filename
from fe.state import init_state, reset_ai_state
from fe.ui_helpers import plotly_chart, dataframe

# Optional css
try:
    from md_utils import inject_css
except Exception:
    def inject_css():
        return


st.set_page_config(page_title="Population Dashboard", layout="wide")
inject_css()

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").strip()

init_state()

st.title("Phân tích dân số theo tỉ lệ sinh - tử")
st.caption(f"Backend: {BACKEND_URL}")

# ---------------- Sidebar ----------------
st.sidebar.header("Tùy chọn")

try:
    countries = get_countries(BACKEND_URL)
except Exception as e:
    st.error(f"Không gọi được backend /worldbank/countries: {e}")
    st.stop()

df_countries = pd.DataFrame(countries)

names_list = df_countries["name"].tolist() if "name" in df_countries.columns else []
sel_names = st.sidebar.multiselect(
    "Chọn quốc gia (tối đa 2):",
    names_list,
    default=["Viet Nam"] if "Viet Nam" in names_list else [],
)

if len(sel_names) > 2:
    st.sidebar.warning("Chỉ chọn tối đa 2 quốc gia. Tự lấy 2 quốc gia đầu.")
    sel_names = sel_names[:2]

current_year = datetime.now().year
start_year = st.sidebar.slider("Năm bắt đầu:", 1960, current_year - 1, 2015)
end_year = st.sidebar.slider("Năm kết thúc:", start_year + 1, current_year, min(current_year, 2025))

if not sel_names:
    st.warning("Vui lòng chọn ít nhất 1 quốc gia.")
    st.stop()

# Reset AI when filters changed (tránh báo cáo AI bị lệch dữ liệu mới)
filters_sig = (tuple(sel_names), int(start_year), int(end_year))
if st.session_state.filters_sig != filters_sig:
    reset_ai_state()
    st.session_state.filters_sig = filters_sig

# ---------------- Load data (không để 1 nước lỗi làm sập app) ----------------
country_dfs: dict[str, pd.DataFrame] = {}
load_errors: dict[str, str] = {}

with st.spinner("Đang tải dữ liệu..."):
    time.sleep(0.1)
    for name in sel_names:
        row = df_countries.loc[df_countries["name"] == name]
        if row.empty or "id" not in row.columns:
            load_errors[name] = "Không tìm thấy country id."
            country_dfs[name] = pd.DataFrame()
            continue

        cid = str(row.iloc[0]["id"])
        try:
            rows = get_series(BACKEND_URL, cid, int(start_year), int(end_year))
            df = normalize_series_df(pd.DataFrame(rows))
            country_dfs[name] = df
        except Exception as e:
            load_errors[name] = str(e)
            country_dfs[name] = pd.DataFrame()

if load_errors:
    for n, msg in load_errors.items():
        st.error(f"Lỗi lấy dữ liệu {n}: {msg}")

if not any(not df.empty for df in country_dfs.values()):
    st.warning("Không tìm thấy dữ liệu hợp lệ.")
    st.stop()

# ---------------- Chart ----------------
st.subheader("So sánh tỉ lệ sinh và tử")

plot_df = pd.concat(
    [df.assign(country=name) for name, df in country_dfs.items() if not df.empty],
    ignore_index=True,
)

if not plot_df.empty and "year" in plot_df.columns:
    long_df = plot_df.melt(
        id_vars=["year", "country"],
        value_vars=[c for c in ["birth_rate", "death_rate"] if c in plot_df.columns],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["year"])
else:
    long_df = pd.DataFrame()

if not long_df.empty:
    long_df["metric"] = long_df["metric"].map({"birth_rate": "Tỉ lệ sinh", "death_rate": "Tỉ lệ tử"}).fillna(long_df["metric"])
    fig = px.line(long_df, x="year", y="value", color="country", line_dash="metric", labels={"value": "Tỉ lệ (per 1,000)"})
    plotly_chart(fig)
else:
    st.info("Không đủ dữ liệu để vẽ biểu đồ.")

# ---------------- Year table (đầy đủ theo năm) ----------------
st.subheader("Dữ liệu theo năm")

col_map = {
    "year": "Năm",
    "birth_rate": "Tỉ lệ sinh (‰)",
    "death_rate": "Tỉ lệ tử (‰)",
    "population": "Dân số",
}

for name, df in country_dfs.items():
    st.write(f"**{name}**")
    if df.empty:
        st.info("Không có dữ liệu.")
        continue

    cols = [c for c in ["year", "birth_rate", "death_rate", "population"] if c in df.columns]
    show_df = df[cols].copy()
    if "year" in show_df.columns:
        show_df = show_df.sort_values("year")

    if "birth_rate" in show_df.columns:
        show_df["birth_rate"] = pd.to_numeric(show_df["birth_rate"], errors="coerce").round(3)
    if "death_rate" in show_df.columns:
        show_df["death_rate"] = pd.to_numeric(show_df["death_rate"], errors="coerce").round(3)
    if "population" in show_df.columns:
        show_df["population"] = pd.to_numeric(show_df["population"], errors="coerce").astype("Int64")

    dataframe(show_df.rename(columns=col_map))

# ---------------- Summary for AI (không có export tóm tắt) ----------------
st.subheader("Phân tích tự động (tóm tắt để đưa vào AI)")

report_rows = []
for name, df in country_dfs.items():
    if df.empty:
        continue
    b = pd.to_numeric(df.get("birth_rate"), errors="coerce")
    d = pd.to_numeric(df.get("death_rate"), errors="coerce")

    avg_b = safe_mean(b) if b is not None else None
    avg_d = safe_mean(d) if d is not None else None

    report_rows.append({
        "Quốc gia": name,
        "Tỉ lệ sinh TB (‰)": f"{avg_b:.2f}" if avg_b is not None else "N/A",
        "Tỉ lệ tử TB (‰)": f"{avg_d:.2f}" if avg_d is not None else "N/A",
        "Xu hướng tỉ lệ sinh": safe_trend(b) if b is not None else "Không đủ dữ liệu",
    })

if report_rows:
    st.table(pd.DataFrame(report_rows))
else:
    st.info("Không đủ dữ liệu để tạo tóm tắt.")

summary_text = "\n".join([
    f"{r['Quốc gia']}: Sinh TB {r['Tỉ lệ sinh TB (‰)']}‰, Tử TB {r['Tỉ lệ tử TB (‰)']}‰, Xu hướng {r['Xu hướng tỉ lệ sinh']}"
    for r in report_rows
]).strip()

# ---------------- AI ----------------
st.markdown("---")
st.header("Phân tích chuyên sâu (AI) - Gemini")

# Prompt bổ sung: chỉ hiện trước lần phân tích đầu tiên
if not st.session_state.ai_generated and not st.session_state.prompt_locked:
    st.text_area(
        "Prompt bổ sung (tuỳ chọn, chỉ dùng 1 lần trước khi phân tích):",
        key="user_prompt_input",
        placeholder="Ví dụ: Nhấn mạnh năm gần nhất, so sánh 2 nước theo rủi ro, đề xuất chính sách cụ thể...",
        height=120,
    )

analyze_clicked = st.button("Phân tích", type="primary", disabled=st.session_state.ai_generated)

if analyze_clicked:
    if not summary_text:
        st.session_state.last_error = "Không có dữ liệu tóm tắt để gửi."
        st.error(st.session_state.last_error)
    else:
        try:
            user_prompt = (st.session_state.get("user_prompt_input", "") or "").strip()
            st.session_state.ai_prompt_used = user_prompt
            st.session_state.prompt_locked = True

            payload_text = build_ai_input(summary_text, user_prompt)
            with st.spinner("Đang gọi AI..."):
                md = analyze(BACKEND_URL, payload_text)

            md = (md or "").strip()
            if not md:
                st.session_state.last_error = "AI trả về rỗng."
                st.error(st.session_state.last_error)
            else:
                st.session_state.ai_report_md = md
                st.session_state.ai_generated = True
                st.session_state.approved = False
                st.session_state.last_status = "Đã tạo báo cáo."
                st.session_state.last_error = ""
                st.session_state.revise_nonce += 1
                st.rerun()
        except Exception as e:
            st.session_state.last_error = str(e)
            st.error(f"Lỗi phân tích: {e}")

if st.session_state.last_status:
    st.success(st.session_state.last_status)
if st.session_state.last_error:
    st.error(st.session_state.last_error)

# ---------------- Render report (chỉ 1 lần) ----------------
if st.session_state.ai_generated and st.session_state.ai_report_md:
    if st.session_state.ai_prompt_used:
        st.caption("Prompt đã dùng (đã khóa):")
        st.text_area("Prompt đã dùng", value=st.session_state.ai_prompt_used, height=80, disabled=True)

    st.subheader("Báo cáo AI")
    st.markdown(st.session_state.ai_report_md)

    # -------- Revise --------
    st.subheader("Yêu cầu chỉnh sửa")
    revise_key = f"revise_input_{st.session_state.revise_nonce}"

    st.text_area(
        "Nhập yêu cầu chỉnh sửa:",
        key=revise_key,
        placeholder="Ví dụ: Rút gọn phần rủi ro, thêm kết luận súc tích, nhấn mạnh so sánh 2 nước...",
        height=120,
    )

    if st.button("Gửi chỉnh sửa"):
        req = (st.session_state.get(revise_key, "") or "").strip()
        if not req:
            st.warning("Bạn chưa nhập yêu cầu chỉnh sửa.")
        else:
            try:
                with st.spinner("Đang chỉnh sửa..."):
                    new_md = revise(BACKEND_URL, st.session_state.ai_report_md, req, system_prompt=BASE_REVISE_PROMPT)
                new_md = (new_md or "").strip()
                if not new_md:
                    st.error("AI chỉnh sửa trả về rỗng.")
                else:
                    st.session_state.ai_report_md = new_md
                    st.session_state.approved = False
                    st.session_state.last_status = "Đã cập nhật báo cáo mới."
                    st.session_state.last_error = ""
                    st.session_state.revise_nonce += 1
                    st.rerun()
            except Exception as e:
                st.session_state.last_error = str(e)
                st.error(f"Lỗi chỉnh sửa: {e}")

    # -------- Approve + Export --------
    st.subheader("Duyệt báo cáo")
    st.checkbox("Duyệt bài báo cáo", key="approved")

    if st.session_state.approved:
        st.subheader("Tải file (chỉ nội dung AI)")
        export_md = (st.session_state.ai_report_md or "").strip()
        export_text = markdown_to_plain_text(export_md)

        file_base = safe_filename(f"Bao_cao_{'_'.join(sel_names)}_{start_year}_{end_year}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Tải MD", data=export_md, file_name=f"{file_base}.md", mime="text/markdown")
        with c2:
            try:
                pdf_bytes = pdf_bytes_from_text(export_text)
                st.download_button("Tải PDF", data=pdf_bytes, file_name=f"{file_base}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Không xuất được PDF: {e}")
        with c3:
            try:
                docx_bytes = docx_bytes_from_text(export_text)
                st.download_button(
                    "Tải DOCX",
                    data=docx_bytes,
                    file_name=f"{file_base}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            except Exception as e:
                st.error(f"Không xuất được DOCX: {e}")
else:
    st.info("Chưa có báo cáo AI. Bấm 'Phân tích'.")
