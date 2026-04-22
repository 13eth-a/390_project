"""
Utilities for loading a political Facebook ads CSV and converting it into a
stable modeling dataset for Week 2.

Expected output columns after cleaning:
- text
- sponsor
- state
- impressions_mid
- spend_mid
- delivery_start
- split  (train / val / test)
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TEXT_CANDIDATES = [
    "ad_creative_body",
    "ad_text",
    "creative_body",
    "body",
    "text",
    "message",
    "ad_copy",
]

SPONSOR_CANDIDATES = [
    "page_name",
    "sponsor",
    "advertiser_name",
    "byline",
    "page",
]

DATE_CANDIDATES = [
    "ad_delivery_start_time",
    "delivery_start",
    "start_date",
    "ad_creation_time",
    "date",
]

IMPRESSIONS_CANDIDATES = [
    "impressions",
    "impression_range",
    "impressions_range",
    "estimated_audience_size",
]

SPEND_CANDIDATES = [
    "spend",
    "spend_range",
    "amount_spent",
]

STATE_CANDIDATES = [
    "state",
    "region",
    "region_distribution",
    "geo",
    "location",
]


def _find_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def _clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_numeric_bounds(value: object) -> tuple[Optional[float], Optional[float]]:
    """
    Handles values such as:
    - {"lower_bound": "1000", "upper_bound": "4999"}
    - {'lower_bound': '1000', 'upper_bound': '4999'}
    - "1000-4999"
    - "1,000 - 4,999"
    - 1234
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None

    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v, v

    s = str(value).strip()
    if not s:
        return None, None

    if s.startswith("{") and s.endswith("}"):
        parsed = None
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            lb = parsed.get("lower_bound") or parsed.get("lower") or parsed.get("min")
            ub = parsed.get("upper_bound") or parsed.get("upper") or parsed.get("max")
            try:
                lb = float(str(lb).replace(",", "")) if lb is not None else None
                ub = float(str(ub).replace(",", "")) if ub is not None else None
                return lb, ub
            except Exception:
                pass

    nums = re.findall(r"\d[\d,]*\.?\d*", s)
    if not nums:
        return None, None
    nums = [float(n.replace(",", "")) for n in nums]
    if len(nums) == 1:
        return nums[0], nums[0]
    return nums[0], nums[1]


def _midpoint_from_value(value: object) -> Optional[float]:
    lb, ub = _extract_numeric_bounds(value)
    if lb is None and ub is None:
        return None
    if lb is None:
        return ub
    if ub is None:
        return lb
    return (lb + ub) / 2.0


def _extract_state(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return "unknown"
        if s.startswith("[") and s.endswith("]"):
            parsed = None
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    parsed = None
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, dict):
                    for key in ("region", "name", "state"):
                        if key in first and first[key]:
                            return str(first[key]).strip()
        return s
    return str(value)


def _stable_split_key(row: pd.Series, idx: int) -> str:
    raw = f"{idx}|{row.get('text', '')}|{row.get('sponsor', '')}|{row.get('delivery_start', '')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _assign_split(df: pd.DataFrame) -> pd.Series:
    """
    Deterministic 70/15/15 split using MD5 hash rather than RNG.
    """
    buckets = []
    for idx, row in df.iterrows():
        h = _stable_split_key(row, idx)
        bucket = int(h[:8], 16) % 100
        if bucket < 70:
            buckets.append("train")
        elif bucket < 85:
            buckets.append("val")
        else:
            buckets.append("test")
    return pd.Series(buckets, index=df.index, name="split")


def load_and_prepare_ads(
    csv_path: str | Path,
    min_year: Optional[int] = 2022,
    max_year: Optional[int] = 2023,
    min_text_len: int = 20,
    save_processed_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    text_col = _find_first_existing(df, TEXT_CANDIDATES)
    sponsor_col = _find_first_existing(df, SPONSOR_CANDIDATES)
    date_col = _find_first_existing(df, DATE_CANDIDATES)
    imp_col = _find_first_existing(df, IMPRESSIONS_CANDIDATES)
    spend_col = _find_first_existing(df, SPEND_CANDIDATES)
    state_col = _find_first_existing(df, STATE_CANDIDATES)

    if text_col is None:
        raise ValueError(f"Could not find an ad text column. Available columns: {list(df.columns)}")
    if imp_col is None:
        raise ValueError(f"Could not find an impressions column. Available columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["text"] = df[text_col].map(_clean_text)
    out["sponsor"] = df[sponsor_col].astype(str) if sponsor_col else "unknown"
    out["state"] = df[state_col].map(_extract_state) if state_col else "unknown"
    out["impressions_mid"] = df[imp_col].map(_midpoint_from_value)
    out["spend_mid"] = df[spend_col].map(_midpoint_from_value) if spend_col else np.nan

    if date_col:
        out["delivery_start"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        out["delivery_start"] = pd.NaT

    out = out.dropna(subset=["impressions_mid"])
    out = out[out["text"].str.len() >= min_text_len].copy()

    if date_col and min_year is not None and max_year is not None:
        year_mask = out["delivery_start"].dt.year.between(min_year, max_year, inclusive="both")
        out = out[year_mask].copy()

    out["log_impressions_mid"] = np.log1p(out["impressions_mid"])
    out["split"] = _assign_split(out)

    keep_cols = [
        "text",
        "sponsor",
        "state",
        "impressions_mid",
        "log_impressions_mid",
        "spend_mid",
        "delivery_start",
        "split",
    ]
    out = out[keep_cols].reset_index(drop=True)

    if save_processed_path is not None:
        save_processed_path = Path(save_processed_path)
        save_processed_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_processed_path, index=False)

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and clean political ads CSV.")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path to save processed CSV")
    parser.add_argument("--min_year", type=int, default=2022)
    parser.add_argument("--max_year", type=int, default=2023)
    args = parser.parse_args()

    cleaned = load_and_prepare_ads(
        csv_path=args.input,
        min_year=args.min_year,
        max_year=args.max_year,
        save_processed_path=args.output,
    )
    print(f"Saved cleaned dataset with {len(cleaned)} rows to {args.output}")
    print(cleaned.head(3).to_string(index=False))