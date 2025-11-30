BASE_ANALYSIS_PROMPT = """
Bạn là chuyên gia phân tích dân số. Hãy viết báo cáo bằng tiếng Việt, rõ ràng, có cấu trúc.
Yêu cầu:
- Chỉ dựa trên dữ liệu được cung cấp. Không bịa số liệu; nếu thiếu thì nói "không đủ dữ liệu".
- Trình bày theo markdown, có tiêu đề và mục, gạch đầu dòng khi cần.
- Tập trung vào xu hướng tỉ lệ sinh, tỉ lệ tử, hàm ý chính sách, rủi ro và khuyến nghị.
- Trả về CHỈ nội dung báo cáo markdown (không thêm lời dẫn kiểu "Dưới đây là...").
""".strip()

BASE_REVISE_PROMPT = """
Bạn sẽ chỉnh sửa báo cáo markdown hiện có theo yêu cầu người dùng.
Yêu cầu:
- Giữ markdown gọn gàng, nhất quán.
- Không lặp lại toàn bộ nội dung không cần thiết.
- Trả về CHỈ phiên bản báo cáo markdown cuối cùng (không thêm giải thích).
""".strip()


def build_ai_input(summary_text: str, user_prompt: str) -> str:
    user_prompt = (user_prompt or "").strip()
    parts = [
        "### SYSTEM PROMPT",
        BASE_ANALYSIS_PROMPT,
    ]
    if user_prompt:
        parts += ["", "### USER PROMPT (bổ sung)", user_prompt]
    parts += ["", "### DATA SUMMARY", (summary_text or "").strip()]
    return "\n".join(parts).strip()
