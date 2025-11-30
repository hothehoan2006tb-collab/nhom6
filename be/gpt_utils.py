# be/gpt_utils.py
import os
import textwrap

import google.generativeai as genai


def _get_model():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment (.env)")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    return genai.GenerativeModel(model_name)


BASE_PROMPT_ANALYZE = textwrap.dedent("""
Bạn là chuyên gia phân tích dân số. Hãy viết một báo cáo chi tiết bằng tiếng Việt dựa trên tóm tắt dữ liệu.

Yêu cầu bắt buộc:
- Trả về Markdown chuẩn (không bọc trong ```).
- Có đề mục rõ ràng:
  1) Giới thiệu
  2) Phân tích xu hướng
  3) Lợi ích
  4) Tác hại / Rủi ro
  5) Biện pháp cần làm ngay
  6) Dự kiến biện pháp 1–10 năm tới
  7) Kết luận ngắn gọn
- Viết mạch lạc, thực tế, có gạch đầu dòng khi phù hợp.
""").strip()


BASE_PROMPT_REVISE = textwrap.dedent("""
Bạn là biên tập viên/chuyên gia dân số. Nhiệm vụ: chỉnh sửa báo cáo theo yêu cầu.

Quy tắc:
- Trả về Markdown chuẩn (không bọc trong ```).
- Viết lại TOÀN BỘ báo cáo (không chỉ trả lời ngắn).
- Giữ cấu trúc đề mục rõ ràng, nội dung nhất quán, tránh lan man.
""").strip()


def gemini_analyze_summary(summary_text: str, user_prompt: str = "") -> str:
    model = _get_model()

    extra = user_prompt.strip()
    prompt = (
        f"{BASE_PROMPT_ANALYZE}\n\n"
        f"## Tóm tắt dữ liệu\n{summary_text.strip()}\n\n"
        + (f"## Yêu cầu bổ sung của người dùng\n{extra}\n\n" if extra else "")
        + "Hãy viết báo cáo ngay bây giờ."
    )

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def gemini_revise_report(report_markdown: str, edit_request: str) -> str:
    model = _get_model()

    prompt = (
        f"{BASE_PROMPT_REVISE}\n\n"
        f"## Báo cáo gốc (Markdown)\n{report_markdown.strip()}\n\n"
        f"## Yêu cầu chỉnh sửa\n{edit_request.strip()}\n\n"
        "Hãy trả về bản báo cáo mới (Markdown) sau khi đã chỉnh sửa."
    )

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()
