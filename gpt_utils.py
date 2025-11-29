# gpt_utils.py
import openai
import pandas as pd

def validate_openai_key(api_key: str):
    try:
        openai.api_key = api_key
        openai.models.list()
        return True, "OK"
    except Exception as e:
        return False, str(e)

def query_insight_with_gpt(api_key: str, country_name: str, df: pd.DataFrame):
    openai.api_key = api_key
    data_text = df.to_csv(index=False)
    prompt = f"""
    Bạn là chuyên gia phân tích dân số. Hãy viết một báo cáo chi tiết bằng tiếng Việt
    dựa trên dữ liệu tỉ lệ sinh (birth_rate) và tỉ lệ tử (death_rate) của quốc gia {country_name}.

    Dữ liệu:
    {data_text}

    Báo cáo cần có cấu trúc rõ ràng:
    1️⃣ Giới thiệu
    2️⃣ Phân tích xu hướng
    3️⃣ Lợi ích
    4️⃣ Tác hại / Rủi ro
    5️⃣ Biện pháp cần làm ngay
    6️⃣ Dự kiến biện pháp 1–10 năm tới
    7️⃣ Kết luận ngắn gọn

    Viết mạch lạc, có dấu, dạng bài báo cáo thực tế.
    """
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Không thể phân tích bằng GPT: {e}"
