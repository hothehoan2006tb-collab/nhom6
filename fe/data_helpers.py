import math
import pandas as pd


def normalize_series_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá DataFrame series từ backend:
    - year -> Int64
    - birth_rate, death_rate, population -> numeric (NaN nếu lỗi)
    - sort theo year
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    for c in ["birth_rate", "death_rate", "population"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "year" in out.columns:
        out = out.sort_values("year")

    return out


def safe_mean(s: pd.Series) -> float | None:
    s2 = pd.to_numeric(s, errors="coerce")
    s2 = s2.replace([math.inf, -math.inf], pd.NA).dropna()
    if s2.empty:
        return None
    return float(s2.mean())


def safe_trend(s: pd.Series) -> str:
    s2 = pd.to_numeric(s, errors="coerce")
    s2 = s2.replace([math.inf, -math.inf], pd.NA).dropna()
    if len(s2) < 2:
        return "Không đủ dữ liệu"
    return "Giảm" if s2.iloc[-1] < s2.iloc[0] else "Tăng"
