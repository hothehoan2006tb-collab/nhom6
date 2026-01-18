import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from dotenv import load_dotenv

from fe.api_client import get_countries, get_series
from fe.data_helpers import normalize_series_df
from fe.export_ai import markdown_to_plain_text, pdf_bytes_from_text, docx_bytes_from_text, safe_filename
from fe.state import init_state, reset_ai_state

# Optional css
try:
    from md_utils import inject_css
except Exception:
    def inject_css():
        return


st.set_page_config(page_title="Population Analysis Dashboard", layout="wide")
inject_css()

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001").strip()

init_state()

st.title("📊 Phân tích Dân Số - Lập Trình Khoa Học Dữ Liệu")
st.caption(f"Backend: {BACKEND_URL} | **Logic xử lý THỦ CÔNG - Không phụ thuộc AI**")

# ============== SIDEBAR ==============
st.sidebar.header("⚙️ Tùy chọn")

try:
    countries = get_countries(BACKEND_URL)
except Exception as e:
    st.error(f"❌ Không kết nối được backend: {e}")
    st.stop()

df_countries = pd.DataFrame(countries)

names_list = df_countries["name"].tolist() if "name" in df_countries.columns else []
sel_names = st.sidebar.multiselect(
    "Chọn quốc gia (tối đa 2):",
    names_list,
    default=["Viet Nam"] if "Viet Nam" in names_list else [],
)

if len(sel_names) > 2:
    st.sidebar.warning("⚠️ Chỉ chọn tối đa 2 quốc gia. Tự lấy 2 quốc gia đầu.")
    sel_names = sel_names[:2]

current_year = datetime.now().year
start_year = st.sidebar.slider("Năm bắt đầu:", 1960, current_year - 1, 2015)
end_year = st.sidebar.slider("Năm kết thúc:", start_year + 1, current_year, min(current_year, 2025))

if not sel_names:
    st.warning("⚠️ Vui lòng chọn ít nhất 1 quốc gia.")
    st.stop()

# Reset AI when filters changed
filters_sig = (tuple(sel_names), int(start_year), int(end_year))
if st.session_state.filters_sig != filters_sig:
    reset_ai_state()
    st.session_state.filters_sig = filters_sig

# ============== LOAD RAW DATA ==============
country_dfs: dict[str, pd.DataFrame] = {}
load_errors: dict[str, str] = {}

with st.spinner("🔄 Đang tải dữ liệu từ World Bank..."):
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
        st.error(f"❌ Lỗi lấy dữ liệu {n}: {msg}")

if not any(not df.empty for df in country_dfs.values()):
    st.warning("⚠️ Không tìm thấy dữ liệu hợp lệ.")
    st.stop()

# ============== VISUALIZATION ==============
st.markdown("---")
st.header("📈 Biểu Đồ Dữ Liệu")

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
    
    long_df["metric"] = long_df["metric"].map({
        "birth_rate": "Tỉ lệ sinh", 
        "death_rate": "Tỉ lệ tử"
    }).fillna(long_df["metric"])
    
    fig = px.line(
        long_df, 
        x="year", 
        y="value", 
        color="country", 
        line_dash="metric",
        labels={"value": "Tỉ lệ (‰)", "year": "Năm"}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ℹ️ Không đủ dữ liệu để vẽ biểu đồ.")

# ============== STATISTICAL PROCESSING (CODE THỦ CÔNG) ==============
st.markdown("---")
st.header("🔬 PHÂN TÍCH THỐNG KÊ THỦ CÔNG")
st.caption("Tất cả số liệu được tính bằng CODE PYTHON - Không dùng AI")

# Dictionary để lưu statistics
country_statistics = {}

for name in sel_names:
    if country_dfs.get(name) is None or country_dfs[name].empty:
        continue
    
    st.subheader(f"📊 {name}")
    
    # Get country_id
    row = df_countries.loc[df_countries["name"] == name]
    country_id = str(row.iloc[0]["id"])
    
    try:
        with st.spinner(f"⚙️ Đang xử lý thống kê cho {name}..."):
            response = requests.post(
                f"{BACKEND_URL}/statistics/process",
                params={
                    "country_id": country_id,
                    "start_year": start_year,
                    "end_year": end_year,
                    "country_name": name
                }
            )
            response.raise_for_status()
            result = response.json()
        
        statistics = result['statistics']
        country_statistics[name] = statistics
        
        # === METRICS CARDS ===
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            birth_mean = statistics['birth_rate_analysis']['mean']
            birth_change = statistics['birth_rate_analysis']['total_change']
            st.metric(
                "Tỉ lệ sinh TB",
                f"{birth_mean}‰",
                delta=f"{birth_change:+.2f}‰"
            )
        
        with col2:
            death_mean = statistics['death_rate_analysis']['mean']
            death_change = statistics['death_rate_analysis']['total_change']
            st.metric(
                "Tỉ lệ tử TB",
                f"{death_mean}‰",
                delta=f"{death_change:+.2f}‰"
            )
        
        with col3:
            natural_increase = statistics['demographic_indicators']['natural_increase_rate']
            st.metric("Tăng tự nhiên", f"{natural_increase}‰")
        
        with col4:
            quality_score = result['data_quality'].get('score', 100)
            st.metric("Data Quality", f"{quality_score}/100")
        
        # === DETAILED TABLE ===
        st.write("**📋 Bảng phân tích chi tiết:**")
        summary_df = pd.DataFrame(result['summary_table'])
        st.dataframe(summary_df, use_container_width=True)
        
        # === ADVANCED STATISTICS ===
        with st.expander(f"📈 Phân tích nâng cao - {name}"):
            birth_trend = statistics['trend_analysis']['birth_rate']
            
            st.write("**Xu hướng tỉ lệ sinh (Linear Regression):**")
            st.write(f"- Hướng: **{birth_trend['direction']}**")
            st.write(f"- Độ tin cậy: **{birth_trend['confidence']}**")
            st.write(f"- R²: {birth_trend['r_squared']} (p-value: {birth_trend['p_value']})")
            st.write(f"- Slope: {birth_trend['slope']:.4f} ‰/năm")
            
            # Confidence Interval
            if 'birth_rate_95ci' in statistics['demographic_indicators']:
                ci = statistics['demographic_indicators']['birth_rate_95ci']
                st.write(f"\n**Confidence Interval (95%):**")
                st.write(f"- Mean: {ci['mean']}‰")
                st.write(f"- Khoảng tin cậy: [{ci['lower_bound']}‰, {ci['upper_bound']}‰]")
                st.write(f"- {ci['interpretation']}")
            
            # Correlation
            if 'correlation_analysis' in statistics:
                corr = statistics['correlation_analysis']
                st.write(f"\n**Tương quan (Birth vs Death):**")
                st.write(f"- Pearson r: {corr['pearson_r']} (p={corr['pearson_p_value']})")
                st.write(f"- {corr['interpretation']}")
            
            # Normality Test
            if 'normality_tests' in statistics and statistics['normality_tests']['birth_rate']:
                norm = statistics['normality_tests']['birth_rate']
                st.write(f"\n**Kiểm định phân phối chuẩn (Shapiro-Wilk):**")
                st.write(f"- {norm['interpretation']}")
            
            # Hypothesis Test
            if 'hypothesis_tests' in statistics and statistics['hypothesis_tests']['birth_rate_vs_20']:
                hyp = statistics['hypothesis_tests']['birth_rate_vs_20']
                st.write(f"\n**Kiểm định giả thuyết (vs world avg 20‰):**")
                st.write(f"- {hyp['conclusion']}")
                st.write(f"- {hyp['interpretation']}")
            
            # Predictions
            if 'predictions' in statistics and 'birth_rate_next_5_years' in statistics['predictions']:
                st.write(f"\n**Dự đoán 5 năm tới (Linear Extrapolation):**")
                pred_df = pd.DataFrame(statistics['predictions']['birth_rate_next_5_years'])
                st.dataframe(pred_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Lỗi xử lý thống kê {name}: {e}")

# === COUNTRY COMPARISON ===
if len(country_statistics) == 2:
    st.markdown("---")
    st.subheader("🔄 So sánh 2 quốc gia")
    
    countries_list = list(country_statistics.keys())
    try:
        comparison_response = requests.post(
            f"{BACKEND_URL}/statistics/compare",
            json={
                "country1_stats": country_statistics[countries_list[0]],
                "country2_stats": country_statistics[countries_list[1]]
            }
        )
        comparison_response.raise_for_status()
        comp = comparison_response.json()['comparison']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{countries_list[0]}**")
            st.write(f"- Tỉ lệ sinh TB: {comp['birth_rate_comparison'][f'{countries_list[0]}_mean']}‰")
            st.write(f"- Xu hướng: {comp['trend_comparison'][f'{countries_list[0]}_trend']}")
        
        with col2:
            st.write(f"**{countries_list[1]}**")
            st.write(f"- Tỉ lệ sinh TB: {comp['birth_rate_comparison'][f'{countries_list[1]}_mean']}‰")
            st.write(f"- Xu hướng: {comp['trend_comparison'][f'{countries_list[1]}_trend']}")
        
        st.info(f"📊 Chênh lệch: **{comp['birth_rate_comparison']['difference']}‰** ({comp['birth_rate_comparison']['percent_difference']}%)")
        st.info(f"🏆 Tỉ lệ sinh cao hơn: **{comp['birth_rate_comparison']['higher']}**")
    
    except Exception as e:
        st.error(f"❌ Lỗi so sánh: {e}")

# ============== AI ANALYSIS (OPTIONAL) ==============
st.markdown("---")
st.header("🤖 Phân Tích AI (Tùy chọn)")
st.caption("AI chỉ viết báo cáo dựa trên số liệu đã tính sẵn ở trên")

if not country_statistics:
    st.info("ℹ️ Cần có dữ liệu thống kê trước khi dùng AI")
else:
    # Prompt bổ sung
    if not st.session_state.ai_generated and not st.session_state.prompt_locked:
        st.text_area(
            "Prompt bổ sung cho AI (tuỳ chọn):",
            key="user_prompt_input",
            placeholder="VD: Nhấn mạnh chính sách, so sánh với xu hướng thế giới...",
            height=100
        )
    
    analyze_clicked = st.button("🚀 Phân tích bằng AI", type="primary", disabled=st.session_state.ai_generated)
    
    if analyze_clicked:
        try:
            user_prompt = st.session_state.get("user_prompt_input", "") or ""
            st.session_state.ai_prompt_used = user_prompt
            st.session_state.prompt_locked = True
            
            # Prepare statistics
            combined_stats = {"countries": country_statistics}
            
            with st.spinner("🤖 AI đang phân tích..."):
                ai_response = requests.post(
                    f"{BACKEND_URL}/ai/analyze-with-stats",
                    json={
                        "statistics": combined_stats,
                        "user_prompt": user_prompt
                    }
                )
                ai_response.raise_for_status()
                ai_result = ai_response.json()
            
            md = ai_result['markdown']
            st.session_state.ai_report_md = md
            st.session_state.source_statistics = combined_stats
            st.session_state.ai_generated = True
            st.session_state.approved = False
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Lỗi AI: {e}")

# ============== AI REPORT & VALIDATION ==============
if st.session_state.ai_generated and st.session_state.ai_report_md:
    st.subheader("📝 Báo cáo AI")
    st.markdown(st.session_state.ai_report_md)
    
    # === AI VALIDATION (PHÁT HIỆN BỊA) ===
    st.markdown("---")
    st.subheader("Kiểm Chứng Báo Cáo AI - PHÁT HIỆN AI BỊA")
    st.caption("Logic thủ công kiểm tra AI có bịa nội dung không")
    
    # HIỂN THỊ TIÊU CHÍ ĐÁNH GIÁ
    with st.expander("TIÊU CHÍ HỆ THỐNG SẼ ĐÁNH GIÁ", expanded=False):
        st.markdown("""
        ### Hệ thống kiểm tra 3 khía cạnh:
        
        #### 1. CHÍNH XÁC SỐ LIỆU
        - Trích xuất tất cả số từ báo cáo AI
        - So sánh với ground truth từ code Python
        - Tính error %: `|AI - Actual| / Actual × 100`
        - Error ≤ 5%: VERIFIED | 5-15%: SUSPICIOUS | >15%: HALLUCINATION
        
        #### 2. XU HƯỚNG
        - Đếm keywords: "tăng"/"giảm"/"ổn định"
        - So sánh với Linear Regression slope
        - AI = Actual → ĐÚNG | AI ≠ Actual → SAI
        
        #### 3. MÂU THUẪN
        - Phân loại câu theo metric & direction
        - Tìm contradiction về CÙNG metric
        
        ### CÔNG THỨC ĐIỂM (0-100)
        ```
        Base = (verified / total) × 100
        Penalty = (bịa×15) + (wrong_trend×10) + min(mâu_thuẫn×3, 30)
        Score = max(0, Base - Penalty)
        ```
        
        - **95-100**: PASS | **70-94**: WARNING | **0-69**: FAIL
        """)
    
    try:
        with st.spinner("🔬 Đang kiểm chứng AI..."):
            validation_response = requests.post(
                f"{BACKEND_URL}/validate/ai-report",
                json={
                    "ai_report": st.session_state.ai_report_md,
                    "source_statistics": st.session_state.source_statistics
                }
            )
            validation_response.raise_for_status()
            validation = validation_response.json()
        
        # Display overall score
        score = validation['hallucination_score']
        verdict = validation['verdict']
        verdict_emoji = validation['verdict_emoji']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hallucination Score", f"{score}/100")
        with col2:
            if verdict == "PASS":
                st.success(f"{verdict_emoji} {verdict}")
            elif verdict == "WARNING":
                st.warning(f"{verdict_emoji} {verdict}")
            else:
                st.error(f"{verdict_emoji} {verdict}")
        with col3:
            st.info(validation['message'])
        
        # AUTO-REFINEMENT: Show detailed feedback if score < 95
        if score < 95:
            st.warning(f"⚠️ Score: {score}/100 - Cần cải thiện để đạt 95/100")
            
           # Initialize iteration counter
            if 'refinement_iteration' not in st.session_state:
                st.session_state.refinement_iteration = 0
            
            # Detailed feedback
            with st.expander("📋 CHI TIẾT VẤN ĐỀ CẦN SỬA", expanded=True):
                stat_val = validation['statistics_verification']
                
                if stat_val.get('suspicious', []):
                    st.error("**⚠️ Số liệu SAI LỆCH:**")
                    for item in stat_val['suspicious']:
                        st.write(f"- `{item['stat_name']}`: AI nói **{item['ai_value']}**, thực tế **{item['actual_value']}** (sai {item['error_pct']}%)")
                
                if stat_val.get('hallucinations', []):
                    st.error("**❌ Số liệu BỊA/THIẾU:**")
                    for item in stat_val['hallucinations']:
                        if item.get('ai_value') is None:
                            st.write(f"- `{item['stat_name']}`: THIẾU - cần {item['actual_value']}")
                        else:
                            st.write(f"- `{item['stat_name']}`: AI bịa **{item['ai_value']}**, thực tế **{item['actual_value']}**")
                
                trend_check = validation.get('trend_check', {})
                if not trend_check.get('correct', True):
                    st.error(f"**❌ SAI XU HƯỚNG:** {trend_check.get('verdict', '')}")
                
                contradictions = validation.get('contradictions', [])
                if len(contradictions) > 0:
                    st.warning(f"**💬 MÂU THUẪN ({len(contradictions)} chỗ):**")
                    for c in contradictions[:3]:
                        st.write(f"- {c.get('explanation', '')}")
                    if len(contradictions) > 3:
                        st.write(f"- ... và {len(contradictions)-3} mâu thuẫn khác")
            
            # REGENERATE BUTTON
            st.markdown("---")
            col_r1, col_r2 = st.columns([3, 1])
            with col_r1:
                st.write(f"**🔄 Tự động cải thiện:**")
                st.caption(f"Đã regenerate: {st.session_state.refinement_iteration} lần")
            with col_r2:
                if st.button("🚀 Regenerate", type="primary", key="regen_btn"):
                    with st.spinner("🤖 AI đang sửa..."):
                        try:
                            regen_resp = requests.post(
                                f"{BACKEND_URL}/ai/regenerate-with-feedback",
                                json={
                                    "validation_feedback": validation,
                                    "statistics": st.session_state.source_statistics,
                                    "user_prompt": st.session_state.get('ai_prompt_used', '')
                                }
                            )
                            regen_resp.raise_for_status()
                            result = regen_resp.json()
                            
                            st.session_state.ai_report_md = result['markdown']
                            st.session_state.refinement_iteration += 1
                            st.success(f"✅ Regenerated lần {st.session_state.refinement_iteration}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi: {e}")
        
        else:
            # Score >= 95 - SUCCESS!
            st.success(f"🎉 **CHÚC MỪNG!** Báo cáo đạt {score}/100!")
            st.balloons()
        
        # Recommendations
        st.write("**💡 Đánh giá:**")
        for rec in validation['recommendations']:
            if "✅" in rec:
                st.success(rec)
            elif "⚠️" in rec:
                st.warning(rec)
            else:
                st.error(rec)
        
        # Details
        with st.expander("📊 Chi tiết validation"):
            stat_val = validation['statistics_verification']
            
            st.write(f"**Thống kê:**")
            st.write(f"- Tổng số liệu kiểm tra: {stat_val['total_stats']}")
            st.write(f"- ✓ Verified: {stat_val['correct_stats']}")
            st.write(f"- ⚠️ Suspicious: {stat_val['suspicious_stats']}")
            st.write(f"- ❌ Hallucinations: {stat_val['hallucinated_stats']}")
            
            if stat_val.get('verified', []):
                st.write("\n**✓ Số liệu đúng:**")
                for m in stat_val['verified']:
                    st.write(f"- {m['stat_name']}: AI={m['ai_value']}, Thực={m['actual_value']} (Error: {m['error_pct']}%)")
            
            if stat_val.get('hallucinations', []):
                st.write("\n**❌ AI BỊA:**")
                for h in stat_val['hallucinations']:
                    st.write(f"- {h['stat_name']}: {h.get('reason', 'Unknown')}")
                    if 'ai_value' in h and h['ai_value']:
                        st.write(f"  AI nói: {h['ai_value']}, Thực tế: {h['actual_value']}")
            
            trend_val = validation['trend_validation']
            st.write(f"\n**Xu hướng:** {trend_val['verdict']}")
    
    except Exception as e:
        st.error(f"❌ Lỗi validation: {e}")
    
    # === EXPORT ===
    st.markdown("---")
    st.subheader("💾 Xuất File")
    st.checkbox("Duyệt báo cáo để xuất file", key="approved")
    
    if st.session_state.approved:
        export_md = st.session_state.ai_report_md
        export_text = markdown_to_plain_text(export_md)
        file_base = safe_filename(f"Bao_cao_{'_'.join(sel_names)}_{start_year}_{end_year}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("📄 Tải MD", data=export_md, file_name=f"{file_base}.md", mime="text/markdown")
        with col2:
            try:
                pdf_bytes = pdf_bytes_from_text(export_text)
                st.download_button("📕 Tải PDF", data=pdf_bytes, file_name=f"{file_base}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"❌ Lỗi PDF: {e}")
        with col3:
            try:
                docx_bytes = docx_bytes_from_text(export_text)
                st.download_button("📘 Tải DOCX", data=docx_bytes, file_name=f"{file_base}.docx", 
                                 mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"❌ Lỗi DOCX: {e}")
else:
    st.info("ℹ️ Chưa có báo cáo AI. Bấm 'Phân tích bằng AI' ở trên.")

# === FOOTER ===
st.markdown("---")
st.caption("🎓 Đồ án môn Lập Trình Khoa Học Dữ Liệu | Logic xử lý thủ công - Validation AI")
