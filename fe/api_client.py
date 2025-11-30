# import requests
# import streamlit as st


# @st.cache_data(ttl=60 * 60)
# def get_countries(base_url: str):
#     r = requests.get(f"{base_url}/worldbank/countries", timeout=30)
#     r.raise_for_status()
#     return r.json()


# @st.cache_data(ttl=15 * 60)
# def get_series(base_url: str, country_id: str, start_year: int, end_year: int):
#     r = requests.get(
#         f"{base_url}/worldbank/series/{country_id}",
#         params={"start_year": start_year, "end_year": end_year},
#         timeout=60,
#     )
#     r.raise_for_status()
#     return r.json()


# def analyze(base_url: str, summary_text: str) -> str:
#     r = requests.post(
#         f"{base_url}/ai/analyze",
#         json={"summary_text": summary_text},
#         timeout=180,
#     )
#     if r.status_code != 200:
#         raise RuntimeError(f"{r.status_code} - {r.text}")
#     return (r.json() or {}).get("markdown", "") or ""


# def revise(base_url: str, report_md: str, edit_request: str, system_prompt: str | None = None) -> str:
#     payload_candidates = [
#         {"report_markdown": report_md, "edit_request": edit_request, "system_prompt": system_prompt},
#         {"report_markdown": report_md, "edit_request": edit_request},
#         {"report_text": report_md, "edit_request": edit_request},
#         {"markdown": report_md, "edit_request": edit_request},
#         {"report": report_md, "request": edit_request},
#     ]
#     last_err = None
#     for payload in payload_candidates:
#         if payload.get("system_prompt") is None:
#             payload.pop("system_prompt", None)
#         try:
#             r = requests.post(f"{base_url}/ai/revise", json=payload, timeout=180)
#             if r.status_code == 200:
#                 j = r.json() or {}
#                 return j.get("markdown", "") or j.get("report_markdown", "") or j.get("report", "") or ""
#             last_err = f"{r.status_code} - {r.text}"
#         except Exception as e:
#             last_err = str(e)
#     raise RuntimeError(last_err or "Unknown error calling /ai/revise")










# fe/api_client.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

import requests


DEFAULT_TIMEOUT = 180


def _join(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _raise_http(prefix: str, r: requests.Response) -> None:
    raise RuntimeError(f"{prefix}: {r.status_code} - {r.text}")


@lru_cache(maxsize=32)
def _discover_revise_path(base_url: str) -> Optional[str]:
    """
    FastAPI thường có /openapi.json. Dò ra path POST có chữ 'revise'/'edit'/'rewrite'.
    Nếu không dò được thì trả None.
    """
    try:
        r = requests.get(_join(base_url, "/openapi.json"), timeout=20)
        if r.status_code != 200:
            return None
        spec = r.json() or {}
        paths: Dict[str, Any] = spec.get("paths", {}) or {}
        if not paths:
            return None

        # ưu tiên đúng tên quen thuộc
        if "/ai/revise" in paths and "post" in (paths["/ai/revise"] or {}):
            return "/ai/revise"
        if "/ai/revise/" in paths and "post" in (paths["/ai/revise/"] or {}):
            return "/ai/revise/"

        candidates = []
        for p, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            if "post" not in methods:
                continue
            lp = p.lower()
            if ("revise" in lp) or ("rewrite" in lp) or ("edit" in lp):
                candidates.append(p)

        # ưu tiên các path có /ai/
        candidates.sort(key=lambda x: (0 if "/ai/" in x else 1, len(x)))
        return candidates[0] if candidates else None
    except Exception:
        return None


def get_countries(base_url: str) -> list[dict]:
    r = requests.get(_join(base_url, "/worldbank/countries"), timeout=30)
    r.raise_for_status()
    return r.json()


def get_series(base_url: str, country_id: str, start_year: int, end_year: int) -> list[dict]:
    r = requests.get(
        _join(base_url, f"/worldbank/series/{country_id}"),
        params={"start_year": start_year, "end_year": end_year},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def analyze(base_url: str, summary_text: str) -> str:
    r = requests.post(
        _join(base_url, "/ai/analyze"),
        json={"summary_text": summary_text},
        timeout=DEFAULT_TIMEOUT,
    )
    if r.status_code != 200:
        _raise_http("Analyze failed", r)
    j = r.json() or {}
    return (j.get("markdown") or j.get("report_markdown") or j.get("report") or "").strip()


def revise(base_url: str, report_md: str, edit_request: str, system_prompt: str | None = None) -> str:
    payload: dict = {
        "report_markdown": report_md,
        "edit_request": edit_request,
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt

    # 1) thử path dò từ openapi trước (ổn định nhất)
    discovered = _discover_revise_path(base_url)
    try_paths = []
    if discovered:
        try_paths.append(discovered)

    # 2) fallback các path thường gặp
    try_paths += ["/ai/revise", "/ai/revise/"]

    last_err = None
    for path in try_paths:
        try:
            url = _join(base_url, path)
            r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                j = r.json() or {}
                return (j.get("markdown") or j.get("report_markdown") or j.get("report") or "").strip()

            # nếu 404 thì thử path khác
            last_err = f"{url}: {r.status_code} - {r.text}"
            if r.status_code == 404:
                continue

            # các lỗi khác thì báo luôn
            _raise_http(f"Revise failed ({url})", r)
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(last_err or "Revise failed: no valid endpoint found")
