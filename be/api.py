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


# ========== NEW ENDPOINTS - LOGIC THỦ CÔNG ==========

from datetime import datetime
from be.statistical_processor import StatisticalProcessor
from be.validators import DataQualityValidator
from be.ai_hallucination_detector import AIHallucinationDetector


@app.post("/statistics/process")
def process_statistics(country_id: str, start_year: int, end_year: int, country_name: str):
    """
    XỬ LÝ THỐNG KÊ BẰNG CODE THỦ CÔNG - Endpoint quan trọng nhất
    Thay thế AI trong việc tính toán
    """
    try:
        # Load data từ World Bank
        df = get_series_for_country(country_id, start_year, end_year)
        
        # Validate data quality TRƯỚC khi xử lý
        validator = DataQualityValidator()
        quality_check = validator.validate_data_ranges(df)
        completeness = validator.check_data_completeness(df)
        
        # XỬ LÝ THỐNG KÊ THỦ CÔNG
        processor = StatisticalProcessor()
        statistics = processor.process_country_statistics(df, country_name)
        
        # Generate summary table
        summary_table = processor.generate_summary_table(statistics)
        
        return {
            "statistics": statistics,
            "summary_table": summary_table.to_dict(orient='records'),
            "data_quality": {
                **quality_check,
                **completeness
            },
            "processing_method": "manual_code",  # KHÔNG phải AI
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/statistics/compare")
def compare_countries_stats(req: dict):
    """
    So sánh 2 quốc gia bằng code thủ công
    """
    try:
        country1_stats = req.get("country1_stats", {})
        country2_stats = req.get("country2_stats", {})
        
        if not country1_stats or not country2_stats:
            raise HTTPException(status_code=400, detail="Missing country statistics")
        
        processor = StatisticalProcessor()
        comparison = processor.compare_countries(country1_stats, country2_stats)
        
        return {
            "comparison": comparison,
            "processing_method": "manual_code"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/analyze-with-stats")
def analyze_with_stats(req: dict):
    """
    AI phân tích BASED ON statistics đã xử lý
    AI KHÔNG được tự tính - chỉ nhận số liệu có sẵn
    """
    statistics = req.get('statistics', {})
    user_prompt = req.get('user_prompt', '')
    
    if not statistics:
        raise HTTPException(status_code=400, detail="statistics is required")
    
    # Build prompt với số liệu đã tính
    prompt = _build_ai_prompt_from_statistics(statistics, user_prompt)
    
    try:
        md = gemini_analyze_summary(prompt, user_prompt)
        
        return {
            "markdown": md,
            "source_statistics": statistics,  # Để validate sau
            "processing_method": "ai_analysis"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_ai_prompt_from_statistics(statistics: dict, user_prompt: str = "") -> str:
    """Tạo prompt tối ưu cho AI với số liệu đã tính sẵn"""
    
    # Handle multiple countries
    if "countries" in statistics:
        # Multiple countries case
        countries_data = statistics["countries"]
        
        # Get time period from first country
        first_country_stats = list(countries_data.values())[0]
        data_period = first_country_stats['data_period']
        start_year = data_period['start_year']
        end_year = data_period['end_year']
        
        prompt_parts = [
            f"**KHOẢNG THỜI GIAN: {start_year}-{end_year}**",
            "",
            "⚠️ YÊU CẦU VALIDATION:",
            f"- Phân tích TOÀN BỘ giai đoạn {start_year}-{end_year}",
            "- COPY CHÍNH XÁC số liệu dưới đây (đến 2 chữ số thập phân)",
            "- Dùng ĐÚNG xu hướng từ cột 'Xu hướng'",
            "- CHỈ dùng MỘT từ (tăng/giảm/ổn định) cho mỗi chỉ số",
            "",
            "## DỮ LIỆU ĐÃ XỬ LÝ"
        ]
        
        for country_name, stats in countries_data.items():
            birth = stats["birth_rate_analysis"]
            death = stats["death_rate_analysis"]
            trend = stats["trend_analysis"]
            demo = stats["demographic_indicators"]
            period = stats["data_period"]
            
            # Determine trend direction with explicit Vietnamese term
            birth_trend_vn = "GIẢM" if trend['birth_rate']['direction'] == "decreasing" else (
                "TĂNG" if trend['birth_rate']['direction'] == "increasing" else "ỔN ĐỊNH"
            )
            death_trend_vn = "GIẢM" if trend['death_rate']['direction'] == "decreasing" else (
                "TĂNG" if trend['death_rate']['direction'] == "increasing" else "ỔN ĐỊNH"
            )
            
            prompt_parts.extend([
                f"",
                f"### {country_name} ({period['start_year']}-{period['end_year']})",
                f"",
                f"**📌 TỈ LỆ SINH (BIRTH RATE):**",
                f"```",
                f"Giá trị trung bình:     {birth['mean']}‰        ← COPY CHÍNH XÁC số này",
                f"Năm {period['start_year']}:       {birth['first_value']}‰",
                f"Năm {period['end_year']}:       {birth['last_value']}‰",
                f"Thay đổi:               {birth['total_change']}‰ ({birth['percent_change']}%)",
                f"Xu hướng:               {birth_trend_vn}         ← CHỈ dùng từ này!",
                f"Độ tin cậy:             R²={trend['birth_rate']['r_squared']}, p={trend['birth_rate']['p_value']}",
                f"```",
                f"",
                f"**📌 TỈ LỆ TỬ (DEATH RATE):**",
                f"```",
                f"Giá trị trung bình:     {death['mean']}‰        ← COPY CHÍNH XÁC số này",
                f"Thay đổi:               {death['total_change']}‰ ({death['percent_change']}%)",
                f"Xu hướng:               {death_trend_vn}         ← CHỈ dùng từ này!",
                f"```",
                f"",
                f"**📌 CHỈ SỐ KHÁC:**",
                f"```",
                f"Tăng tự nhiên:          {demo['natural_increase_rate']}‰",
                f"Giai đoạn dân số:       {demo['demographic_stage']}",
                f"```"
            ])
    else:
        # Single country
        country = statistics['country']
        birth = statistics['birth_rate_analysis']
        death = statistics['death_rate_analysis']
        trend = statistics['trend_analysis']
        demo = statistics['demographic_indicators']
        period = statistics['data_period']
        
        # Explicit Vietnamese trend terms
        birth_trend_vn = "GIẢM" if trend['birth_rate']['direction'] == "decreasing" else (
            "TĂNG" if trend['birth_rate']['direction'] == "increasing" else "ỔN ĐỊNH"
        )
        death_trend_vn = "GIẢM" if trend['death_rate']['direction'] == "decreasing" else (
            "TĂNG" if trend['death_rate']['direction'] == "increasing" else "ỔN ĐỊNH"
        )
        
        prompt_parts = [
            f"**KHOẢNG THỜI GIAN: {period['start_year']}-{period['end_year']}**",
            "",
            "⚠️ YÊU CẦU VALIDATION:",
            f"- Phân tích TOÀN BỘ giai đoạn {period['start_year']}-{period['end_year']}",
            "- COPY CHÍNH XÁC số liệu dưới đây",
            "- Dùng ĐÚNG xu hướng từ cột 'Xu hướng'",
            "- KHÔNG tự mâu thuẫn",
            "",
            f"## DỮ LIỆU - {country} ({period['start_year']}-{period['end_year']})",
            f"",
            f"**📌 TỈ LỆ SINH (BIRTH RATE):**",
            f"```",
            f"Trung bình giai đoạn:   {birth['mean']}‰        ← COPY số này",
            f"Năm {period['start_year']}:       {birth['first_value']}‰",
            f"Năm {period['end_year']}:       {birth['last_value']}‰",
            f"Thay đổi:               {birth['total_change']}‰ ({birth['percent_change']}%)",
            f"**XU HƯỚNG: {birth_trend_vn}**   ← BẮT BUỘC dùng từ này!",
            f"R²:                     {trend['birth_rate']['r_squared']}",
            f"```",
            f"",
            f"**📌 TỈ LỆ TỬ (DEATH RATE):**",
            f"```",
            f"Trung bình:             {death['mean']}‰",
            f"Thay đổi:               {death['total_change']}‰",
            f"**XU HƯỚNG: {death_trend_vn}**   ← BẮT BUỘC dùng từ này!",
            f"```",
            f"",
            f"**📌 CHỈ SỐ KHÁC:**",
            f"- Tăng tự nhiên: {demo['natural_increase_rate']}‰",
            f"- Giai đoạn dân số: {demo['demographic_stage']}"
        ]
    
    if user_prompt:
        prompt_parts.extend([
            "",
            "## YÊU CẦU BỔ SUNG CỦA NGƯỜI DÙNG",
            user_prompt
        ])
    
    prompt_parts.extend([
        "",
        "---",
        "💡 REMINDER: Validation sẽ kiểm tra:",
        "1. Số liệu có khớp không (tolerance 5%)",
        "2. Xu hướng có đúng không",
        "3. Có tự mâu thuẫn không",
        "",
        "Viết báo cáo MARKDOWN, KHÔNG code block."
    ])
    
    return "\n".join(prompt_parts)


@app.post("/ai/regenerate-with-feedback")
def regenerate_with_feedback(req: dict):
    """
    REGENERATE AI REPORT với feedback từ validation
    Auto-refinement loop cho đến khi đạt 95/100
    """
    validation_feedback = req.get('validation_feedback', {})
    statistics = req.get('statistics', {})
    user_prompt = req.get('user_prompt', '')
    
    if not statistics:
        raise HTTPException(status_code=400, detail="statistics required")
    
    # Build detailed feedback
    feedback_parts = ["## FEEDBACK TỪ HỆ THỐNG VALIDATION\n"]
    score = validation_feedback.get('hallucination_score', 0)
    verdict = validation_feedback.get('verdict', 'UNKNOWN')
    feedback_parts.append(f"**Điểm hiện tại:** {score}/100 ({verdict})\n")
    
    # Statistics verification issues
    stats_verif = validation_feedback.get('statistics_verification', {})
    if stats_verif.get('suspicious', []):
        feedback_parts.append("**⚠️ Các số liệu SAI LỆCH cần sửa:**")
        for item in stats_verif['suspicious']:
            feedback_parts.append(
                f"- {item['stat_name']}: AI nói {item['ai_value']}, "
                f"thực tế là {item['actual_value']} (sai lệch {item['error_pct']}%)"
            )
        feedback_parts.append("")
    
    if stats_verif.get('hallucinations', []):
        feedback_parts.append("**❌ Các số liệu BỊA/THIẾU:**")
        for item in stats_verif['hallucinations']:
            if item.get('ai_value') is None:
                feedback_parts.append(f"- {item['stat_name']}: THIẾU - Cần thêm {item['actual_value']}")
            else:
                feedback_parts.append(
                    f"- {item['stat_name']}: AI bịa {item['ai_value']}, "
                    f"thực tế là {item['actual_value']}"
                )
        feedback_parts.append("")
    
    # Trend issues
    trend_check = validation_feedback.get('trend_check', {})
    if not trend_check.get('correct', True):
        feedback_parts.append("**❌ SAI XU HƯỚNG:**")
        feedback_parts.append(f"- {trend_check.get('verdict', '')}")
        feedback_parts.append("")
    
    # Contradictions
    contradictions = validation_feedback.get('contradictions', [])
    if len(contradictions) > 0:
        feedback_parts.append(f"**⚠️ MÂU THUẪN NỘI BỘ ({len(contradictions)} chỗ):**")
        for c in contradictions[:5]:
            feedback_parts.append(f"- {c.get('explanation', 'Unknown')}")
        if len(contradictions) > 5:
            feedback_parts.append(f"- ... và {len(contradictions) - 5} mâu thuẫn khác")
        feedback_parts.append("")
    
    # Recommendations
    recommendations = validation_feedback.get('recommendations', [])
    if recommendations:
        feedback_parts.append("**📋 HƯỚNG DẪN SỬA:**")
        for rec in recommendations:
            feedback_parts.append(f"- {rec}")
        feedback_parts.append("")
    
    feedback_text = "\n".join(feedback_parts)
    
    # Build regeneration prompt
    base_prompt = _build_ai_prompt_from_statistics(statistics, user_prompt)
    
    regeneration_prompt = f"""{base_prompt}

## YÊU CẦU REGENERATE

Báo cáo trước có các vấn đề sau:

{feedback_text}

**NHIỆM VỤ:**
1. Viết LẠI toàn bộ báo cáo
2. SỬA tất cả các số liệu sai lệch/bịa/thiếu
3. SỬA xu hướng nếu sai  
4. LOẠI BỎ mâu thuẫn nội bộ
5. ĐẢM BẢO dùng CHÍNH XÁC số liệu đã cho
6. ĐẢM BẢO nói rõ khoảng thời gian khi phân tích

Trả về MARKDOWN thuần túy, KHÔNG nhắc lại vấn đề đã sửa.
"""
    
    try:
        md = gemini_analyze_summary(regeneration_prompt, "")
        
        return {
            "markdown": md,
            "source_statistics": statistics,
            "processing_method": "ai_regeneration_with_feedback",
            "iteration_feedback": feedback_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/ai-report")
def validate_ai_report(req: dict):
    """
    VALIDATE AI REPORT - Phát hiện AI bịa nội dung
    ĐÂY LÀ LOGIC QUAN TRỌNG NHẤT
    """
    ai_report = req.get('ai_report', '')
    source_statistics = req.get('source_statistics', {})
    
    if not ai_report:
        raise HTTPException(status_code=400, detail="ai_report is required")
    if not source_statistics:
        raise HTTPException(status_code=400, detail="source_statistics is required")
    
    try:
        detector = AIHallucinationDetector()
        
        # Extract actual stats để so sánh
        actual_stats = {}
        actual_trend = None
        country_name = None
        
        try:
            # Handle both single and multiple countries
            print(f"[DEBUG] source_statistics keys: {list(source_statistics.keys())}")
            
            if "countries" in source_statistics:
                # Multiple countries - use first one for trend check
                print(f"[DEBUG] Multiple countries mode, countries: {list(source_statistics['countries'].keys())}")
                first_country = list(source_statistics["countries"].keys())[0]
                stats = source_statistics["countries"][first_country]
                country_name = first_country
                print(f"[DEBUG] Selected country: {country_name}")
                print(f"[DEBUG] Stats keys: {list(stats.keys())}")
            else:
                # Single country
                print(f"[DEBUG] Single country mode")
                stats = source_statistics
                country_name = stats.get('country', 'Unknown')
                print(f"[DEBUG] Country name: {country_name}")
                print(f"[DEBUG] Stats keys: {list(stats.keys())}")
            
            print(f"[DEBUG] Extracting actual_stats...")
            actual_stats = {
                "birth_rate_avg": stats['birth_rate_analysis']['mean'],
                "death_rate_avg": stats['death_rate_analysis']['mean'],
                "birth_min": stats['birth_rate_analysis']['min'],
                "birth_max": stats['birth_rate_analysis']['max'],
                "natural_increase": stats['demographic_indicators']['natural_increase_rate']
            }
            print(f"[DEBUG] Actual stats extracted: {list(actual_stats.keys())}")
            
            actual_trend = stats['trend_analysis']['birth_rate']['direction']
            print(f"[DEBUG] Actual trend: {actual_trend}")
        except KeyError as e:
            print(f"[ERROR] KeyError while extracting stats: {e}")
            print(f"[ERROR] Available keys in stats: {list(stats.keys() if 'stats' in locals() else 'stats not defined')}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing statistic in source_statistics: {str(e)}. Available keys: {list(stats.keys() if 'stats' in locals() else source_statistics.keys())}"
            )
        
        # VERIFY STATISTICS
        try:
            verification = detector.verify_ai_statistics(ai_report, actual_stats, tolerance_percent=5.0)
        except Exception as e:
            print(f"[ERROR] verify_ai_statistics failed: {type(e).__name__}: {e}")
            verification = {
                "verified": [],
                "suspicious": [],
                "hallucinations": [],
                "accuracy_score": 0,
                "total_stats": 0,
                "correct_stats": 0,
                "suspicious_stats": 0,
                "hallucinated_stats": 0,
                "error": f"Lỗi validation: {str(e)}"
            }
        
        # CHECK TREND ACCURACY
        try:
            trend_check = detector.check_trend_accuracy(ai_report, actual_trend, country_name)
        except Exception as e:
            print(f"[ERROR] check_trend_accuracy failed: {type(e).__name__}: {e}")
            trend_check = {
                "correct": False,
                "ai_claimed": "unknown",
                "actual": actual_trend,
                "evidence": [],
                "verdict": f"Lỗi kiểm tra xu hướng: {str(e)}",
                "severity": "ERROR"
            }
        
        # DETECT CONTRADICTIONS
        try:
            contradictions = detector.detect_contradictions(ai_report)
        except Exception as e:
            print(f"[ERROR] detect_contradictions failed: {type(e).__name__}: {e}")
            contradictions = []
        
        # GENERATE FULL REPORT
        try:
            validation_report = detector.generate_validation_report(
                verification, trend_check, contradictions
            )
        except Exception as e:
            print(f"[ERROR] generate_validation_report failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Lỗi tạo báo cáo validation: {str(e)}")
        
        return validation_report
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Validation endpoint failed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi validation: {str(e)}")
