"""
Convert Meta Ad Library API JSONL output → a CSV that data_loader.py can ingest.

The API returns these relevant fields for political ads:
  ad_creative_bodies  list[str]    → we take index 0 as the ad text
  page_name           str          → sponsor
  ad_delivery_start_time str       → date
  impressions         dict         → {"lower_bound": "1000", "upper_bound": "4999"}
  spend               dict         → same format
  delivery_by_region  list[dict]   → we extract the top-percentage region as "state"

Column names written to CSV are chosen so that data_loader._find_first_existing()
recognises them without modification:
  - "ad_creative_body"  (matches TEXT_CANDIDATES)
  - "page_name"         (matches SPONSOR_CANDIDATES)
  - "ad_delivery_start_time" (matches DATE_CANDIDATES)
  - "impressions"       (matches IMPRESSIONS_CANDIDATES)
  - "spend"             (matches SPEND_CANDIDATES)
  - "state"             (matches STATE_CANDIDATES)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field-normalisation helpers
# ---------------------------------------------------------------------------

def _first_body(bodies: Any) -> str:
    """Return the first non-empty element of ad_creative_bodies, or ''."""
    if isinstance(bodies, list):
        for item in bodies:
            if item and isinstance(item, str):
                return item
    if isinstance(bodies, str):
        return bodies
    return ""


def _top_region(region_list: Any) -> str:
    """
    delivery_by_region is a list of dicts: [{region: "US-CA", percentage: "0.15"}, ...]
    Return the sub-national code of the highest-percentage region, e.g. "CA".
    """
    if not isinstance(region_list, list) or not region_list:
        return ""
    try:
        top = max(
            region_list,
            key=lambda x: float(x.get("percentage", 0)) if isinstance(x, dict) else 0,
        )
        region = str(top.get("region", "")) if isinstance(top, dict) else ""
        return region.split("-", 1)[1] if "-" in region else region
    except (ValueError, TypeError):
        return ""


def _serialise(value: Any) -> Any:
    """
    Ensure dict/list values are stored as JSON strings in the CSV so that
    data_loader._extract_numeric_bounds() can parse them back correctly.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def load_jsonl(jsonl_path: str | Path) -> pd.DataFrame:
    """Read a JSONL file and return a DataFrame with one row per ad."""
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", lineno, exc)

    if not records:
        raise ValueError(f"No valid records found in {jsonl_path}")

    logger.info("Loaded %d records from %s", len(records), jsonl_path)
    return pd.DataFrame(records)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename / derive columns so the output CSV is compatible with data_loader.py.
    Returns a new DataFrame with the standardised column set.
    """
    out = pd.DataFrame()

    # --- text ---
    if "ad_creative_bodies" in df.columns:
        out["ad_creative_body"] = df["ad_creative_bodies"].apply(_first_body)
    else:
        out["ad_creative_body"] = ""

    # --- sponsor ---
    out["page_name"] = df.get("page_name", pd.Series("", index=df.index)).fillna("")
    out["page_id"] = df.get("page_id", pd.Series("", index=df.index)).fillna("")

    # --- date ---
    out["ad_delivery_start_time"] = (
        df.get("ad_delivery_start_time", pd.Series("", index=df.index)).fillna("")
    )
    out["ad_delivery_stop_time"] = (
        df.get("ad_delivery_stop_time", pd.Series("", index=df.index)).fillna("")
    )
    out["ad_creation_time"] = (
        df.get("ad_creation_time", pd.Series("", index=df.index)).fillna("")
    )

    # --- impressions & spend (keep as JSON string for data_loader) ---
    for col in ("impressions", "spend", "currency"):
        if col in df.columns:
            out[col] = df[col].apply(_serialise)
        else:
            out[col] = ""

    # --- state / region ---
    if "delivery_by_region" in df.columns:
        out["state"] = df["delivery_by_region"].apply(_top_region)
    else:
        out["state"] = ""

    # --- passthrough metadata ---
    for col in ("id", "bylines", "languages", "publisher_platforms",
                "demographic_distribution", "ad_snapshot_url"):
        if col in df.columns:
            out[col] = df[col].apply(_serialise)

    return out


def convert_jsonl_to_csv(
    jsonl_path: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Full pipeline: JSONL → normalised DataFrame → CSV."""
    df_raw = load_jsonl(jsonl_path)
    df_out = normalise_columns(df_raw)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(df_out), output_csv)
    return df_out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert Meta Ad Library JSONL → pipeline-compatible CSV."
    )
    parser.add_argument("--input", required=True, help="Path to .jsonl file")
    parser.add_argument(
        "--output",
        default="data/raw/meta_ads_raw.csv",
        help="Destination CSV path",
    )
    args = parser.parse_args()

    convert_jsonl_to_csv(args.input, args.output)
