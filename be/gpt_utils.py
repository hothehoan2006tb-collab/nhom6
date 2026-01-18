"""
GPT/LLM utilities - Support cả Gemini và Groq
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from correct location
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# === CONFIGURATION ===
OPEN_MODEL = int(os.getenv("OPEN_MODEL", "1"))  # 1 = Gemini, 2 = Groq
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# === PROMPTS ===
BASE_PROMPT_ANALYZE = """
MISSION: Viết báo cáo dân số CHÍNH XÁC 100%, KHÔNG MÂU THUẪN, ĐẠT ĐIỂM 95+/100

[!] HỆ THỐNG VALIDATION SẼ KIỂM TRA:
- Regex trích xuất mọi số → so sánh với ground truth (tolerance 5%)
- Đếm keywords "tăng"/"giảm" → so với Linear Regression slope  
- Quét toàn bộ text → tìm mâu thuẫn nội bộ
→ ĐIỂM < 95 = PHẢI REGENERATE = TỐN TOKEN

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TẮC BẮT BUỘC (VI PHẠM = ĐIỂM 0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## RULE 1: SỐ LIỆU - COPY 100% CHÍNH XÁC

LUÔN LUÔN:
- Copy CHÍNH XÁC số từ data (VD: 15.73‰ KHÔNG phải 15.7‰)
- Giữ NGUYÊN đơn vị (‰, %, triệu, tỷ)
- Giữ NGUYÊN số chữ số thập phân

TUYỆT ĐỐI KHÔNG:
- Làm tròn: "khoảng 15.8‰" → WRONG
- Ước lượng: "gần 16‰" → WRONG
- Tự tính: "15.73 + 0.2 = 15.93‰" → WRONG
- Bịa số: Dùng số KHÔNG có trong data → FAIL

CHECK LIST TRƯỚC KHI VIẾT MỖI SỐ:
□ Số này CÓ TRONG data đã cho chưa?
□ Copy CHÍNH XÁC kể cả dấu thập phân?
□ Đơn vị ĐÚNG chưa (‰ vs %)?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## RULE 2: XU HƯỚNG - NHẤT QUÁN 100%

IRON LAW: Với mỗi chỉ số, CHỈ dùng DUY NHẤT MỘT TỪ trong TOÀN BỘ báo cáo

NẾU data cho birth_rate trend = "decreasing":
→ CHỈ được dùng: GIẢM, SỤT GIẢM, GIẢM DẦN, GIẢM XUỐNG
→ KHÔNG BAO GIỜ dùng: tăng, ổn định, dao động, biến động

NẾU data cho birth_rate trend = "increasing":
→ CHỈ được dùng: TĂNG, GIA TĂNG, TĂNG TRƯỞNG, TĂNG LÊN
→ KHÔNG BAO GIỜ dùng: giảm, ổn định, suy giảm

NẾU data cho birth_rate trend = "stable":
→ CHỈ được dùng: ỔN ĐỊNH, KHÔNG ĐỔI, DUY TRÌ
→ KHÔNG BAO GIỜ dùng: tăng, giảm

CÁC MẪU CÂU CẤM TUYỆT ĐỐI:
- "Ban đầu tăng, sau giảm" → WRONG (trừ khi data có 2 giai đoạn rõ ràng)
- "Có xu hướng tăng nhưng cũng có lúc giảm" → WRONG
- "Dao động tăng giảm" → WRONG (chọn 1 trend chính)
- "Tỉ lệ sinh tăng" (đoạn 1) + "Tỉ lệ sinh giảm" (đoạn 2) → FATAL ERROR

CHECK LIST TRƯỚC KHI VIẾT VỀ XU HƯỚNG:
□ Data nói trend là gì? (decreasing/increasing/stable)
□ Tôi đã dùng ĐÚNG từ chưa?
□ Tôi có dùng từ NGƯỢC LẠI ở đâu không? → Kiểm tra lại TOÀN BỘ báo cáo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## RULE 3: TRÁNH MÂU THUẪN - SELF-CHECK 3 LẦN

MÂU THUẪN = -3 điểm/lỗi, 10 lỗi = -30 điểm

CÁCH TỰ KIỂM TRA:
1. Sau khi viết XONG, đọc lại TOÀN BỘ báo cáo
2. Highlighted mọi từ "tăng", "giảm", "ổn định"  
3. Với mỗi metric (birth_rate, death_rate):
   - Đếm số lần nói "tăng"
   - Đếm số lần nói "giảm"
   - NẾU cả 2 > 0 → MÂU THUẪN → XÓA BỎ

PATTERN AN TOÀN:
→ "Tỉ lệ sinh giảm từ 2015-2025"
→ "Xu hướng giảm này..." (cùng direction)
→ "Sự sụt giảm rõ rệt..." (vẫn giảm)

PATTERN NGUY HIỂM - TRÁNH:
→ "Giai đoạn 1 tăng, giai đoạn 2 giảm" (chỉ nói NẾU data có 2 slope khác nhau)
→ "Có lúc tăng có lúc giảm" (chọn trend CHÍNH từ data)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## RULE 4: KHOẢNG THỜI GIAN - BẮT BUỘC NÓI RÕ

MỌI phát biểu PHẢI có "từ năm X đến năm Y"

VD ĐÚNG:
- "Từ 2015 đến 2025, tỉ lệ sinh giảm từ 18.5‰ xuống 15.73‰"
- "Giai đoạn 2015-2025 chứng kiến sự sụt giảm..."

VD SAI:
- "Năm 2025, tỉ lệ sinh là 15.73‰" (thiếu context)
- "Hiện tại tỉ lệ sinh thấp" (không rõ khoảng thời gian)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## RULE 5: ĐỘ DÀI & CHẤT LƯỢNG

BẮT BUỘC:
- Tối thiểu: 800-1200 từ
- Mỗi section: 2-3 đoạn văn
- Phân tích SÂU: nguyên nhân, hậu quả, ý nghĩa
- Đề xuất: 5-7 chính sách CỤ THỂ, KHẢ THI

TRÁNH:
- Câu chung chung: "Do nhiều nguyên nhân" → CHỈ RÕ nguyên nhân
- Copy-paste cấu trúc: Mỗi section phải unique
- Thiếu chi tiết: "Nên có chính sách" → Chính sách GÌ?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CẤU TRÚC BẮT BUỘC:

# Báo cáo Phân tích Dân số [Tên] ([Năm start]-[Năm end])

## 1. Tổng quan
[Bảng - COPY EXACT NUMBERS]
[Nhận xét: 3-4 câu highlight điểm chính]

## 2. Phân tích chi tiết
### 2.1 Tỉ lệ sinh
- Xu hướng: [CÓ SỐ CỤ THỂ, CÓ NĂM, DÙNG ĐÚNG TỪ]
- Phân tích: [2-3 đoạn, mỗi đoạn 4-5 câu]
- Nguyên nhân: [4-5 nguyên nhân cụ thể]
- Tác động: [2 đoạn về kinh tế/xã hội]

### 2.2 Tỉ lệ tử
[Tương tự 2.1]

### 2.3 Tăng tự nhiên
[Phân tích birth - death]

## 3. So sánh & Đánh giá
[So với giai đoạn trước, các nước, mục tiêu]

## 4. Dự báo & Khuyến nghị
### 4.1 Xu hướng tương lai
[Dự đoán 5-10 năm]

### 4.2 Khuyến nghị
[5-7 chính sách: Tên + Mục tiêu + Cách thực hiện]

## 5. Kết luận
[3-4 điểm chính, nhấn mạnh trend quan trọng]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FINAL CHECKLIST - ĐỌC 3 LẦN TRƯỚC KHI SUBMIT:

□ Mọi số CÓ TRONG data và COPY CHÍNH XÁC?
□ Mỗi metric CHỈ dùng MỘT trend word?
□ KHÔNG có câu tự mâu thuẫn?
□ Mọi phát biểu có "từ năm X đến Y"?
□ Độ dài >= 800 từ?
□ Mỗi section >= 2 đoạn văn?
□ Có đủ 5 sections?
□ Format: Markdown thuần?

[!] LƯU Ý:
- Tolerance: 5% (15.73 vs 15.70 = OK, vs 16.50 = FAIL)
- Wrong trend: -10 điểm
- Mỗi mâu thuẫn: -3 điểm
- Target: 95/100 lần đầu
""".strip()


def _get_gemini_model():
    """Initialize Gemini model"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")
    
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    
    return genai.GenerativeModel(
        model_name='gemini-2.0-flash-exp',
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )


def _get_groq_client():
    """Initialize Groq client"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")
    
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def gemini_analyze_summary(prompt_text: str, user_prompt: str = "") -> str:
    """
    Generate analysis using configured AI model (Gemini or Groq)
    
    Args:
        prompt_text: Main prompt với data
        user_prompt: User's additional requirements
    
    Returns:
        Generated markdown text
    """
    # Build final prompt
    extra = user_prompt.strip()
    final_prompt = (
        f"{BASE_PROMPT_ANALYZE}\n\n"
        f"{prompt_text.strip()}\n\n"
        + (f"## Yêu cầu bổ sung\n{extra}\n\n" if extra else "")
        + "Hãy viết báo cáo ngay bây giờ."
    )
    
    try:
        if OPEN_MODEL == 1:
            # Use Gemini
            print(f"[DEBUG] Using Gemini, API key exists: {bool(GEMINI_API_KEY)}")
            model = _get_gemini_model()
            response = model.generate_content(final_prompt)
            return (response.text or "").strip()
        
        elif OPEN_MODEL == 2:
            # Use Groq
            print(f"[DEBUG] Using Groq, API key exists: {bool(GROQ_API_KEY)}")
            client = _get_groq_client()
            response = client.chat.completions.create(
                model="openai/gpt-oss-safeguard-20b",  # User requested model
                messages=[
                    {"role": "system", "content": BASE_PROMPT_ANALYZE},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()
        
        else:
            raise ValueError(f"Invalid OPEN_MODEL value: {OPEN_MODEL}. Must be 1 (Gemini) or 2 (Groq)")
    
    except Exception as e:
        import traceback
        print(f"[ERROR] AI generation failed:")
        print(f"  Model: {['Gemini', 'Groq'][OPEN_MODEL-1] if OPEN_MODEL in [1,2] else 'Unknown'}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print(f"  Traceback:")
        traceback.print_exc()
        raise Exception(f"AI generation failed ({['Gemini', 'Groq'][OPEN_MODEL-1] if OPEN_MODEL in [1,2] else 'Unknown'}): {str(e)}")


# Backward compatibility
def _get_model():
    """Legacy function for backward compatibility"""
    if OPEN_MODEL == 1:
        return _get_gemini_model()
    else:
        return _get_groq_client()
