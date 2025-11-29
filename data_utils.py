# data_utils.py
import requests
import pandas as pd

WB_BASE = "http://api.worldbank.org/v2"

def get_country_list_worldbank():
    """Trả về list dict: {id (ISO3), name, iso2Code}"""
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
            year = int(item.get("date"))
            value = item.get("value")
            if value is not None:
                series[year] = float(value)
    return series

def get_birth_death_series_worldbank(country_id, start_year, end_year):
    """
    Trả DataFrame columns: year, birth_rate, death_rate
    World Bank indicator codes:
      - Crude birth rate: SP.DYN.CBRT.IN
      - Crude death rate: SP.DYN.CDRT.IN
    Values are per 1,000 population (crude rates)
    """
    birth_ind = "SP.DYN.CBRT.IN"
    death_ind = "SP.DYN.CDRT.IN"
    b = _fetch_indicator(country_id, birth_ind, start_year, end_year)
    d = _fetch_indicator(country_id, death_ind, start_year, end_year)
    years = sorted(set(list(b.keys()) + list(d.keys())))
    rows = []
    for y in years:
        rows.append({
            "year": y,
            "birth_rate": b.get(y),
            "death_rate": d.get(y),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("year")
    return df
