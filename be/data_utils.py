import requests
import pandas as pd

WB_BASE = "http://api.worldbank.org/v2"

def get_country_list_worldbank():
    url = f"{WB_BASE}/country?format=json&per_page=500"
    res = requests.get(url, timeout=10)
    data = res.json()
    countries = []
    if isinstance(data, list) and len(data) >= 2:
        for c in data[1]:
            countries.append({"id": c.get("id"), "name": c.get("name"), "iso2Code": c.get("iso2Code")})
    return countries

def _fetch_indicator(country_id, indicator, start_year, end_year):
    url = f"{WB_BASE}/country/{country_id}/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=1000"
    res = requests.get(url, timeout=15)
    data = res.json()
    series = {}
    if isinstance(data, list) and len(data) >= 2:
        for item in data[1]:
            try:
                year = int(item.get("date"))
                value = item.get("value")
                if value is not None:
                    series[year] = float(value)
            except Exception:
                continue
    return series

def get_series_for_country(country_id, start_year, end_year):
    birth_ind = "SP.DYN.CBRT.IN"
    death_ind = "SP.DYN.CDRT.IN"
    pop_ind = "SP.POP.TOTL"

    b = _fetch_indicator(country_id, birth_ind, start_year, end_year)
    d = _fetch_indicator(country_id, death_ind, start_year, end_year)
    p = _fetch_indicator(country_id, pop_ind, start_year, end_year)

    years = sorted(set(b) | set(d) | set(p))
    rows = [{"year": y, "birth_rate": b.get(y), "death_rate": d.get(y), "population": p.get(y)} for y in years]
    return pd.DataFrame(rows).sort_values("year") if rows else pd.DataFrame()
