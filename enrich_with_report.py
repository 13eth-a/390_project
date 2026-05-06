"""
Enrich the processed FBPAC ads dataset with advertiser-level features
from the Meta Ad Library bulk report (FacebookAdLibraryReport_*_advertisers.csv).

Join key: normalized page_name (lowercase, stripped) on both sides.

New columns added to processed CSV:
  - adv_total_spend_usd   : lifetime USD spend for this advertiser
  - adv_total_ads         : lifetime ad count for this advertiser
  - adv_log_spend         : log(1 + adv_total_spend_usd)
  - adv_log_ads           : log(1 + adv_total_ads)

Usage:
  python enrich_with_report.py \
    --report ~/Downloads/FacebookAdLibraryReport_2026-04-26_US_lifelong/FacebookAdLibraryReport_2026-04-26_US_lifelong_advertisers.csv \
    --processed data/processed/political_ads_processed.csv \
    --output data/processed/political_ads_enriched.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def normalise_name(s: object) -> str:
    """Lowercase + strip for fuzzy sponsor matching."""
    return str(s).lower().strip() if pd.notna(s) else ""


def load_report(report_path: Path) -> pd.DataFrame:
    # The report CSV has a multi-line header — skip the first row (title line)
    # and read "Page name", "Amount spent (USD)", "Number of ads in Library"
    df = pd.read_csv(report_path, skiprows=0, on_bad_lines="skip")
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Detect the right column names (Meta has changed these across report versions)
    name_col = next(
        (c for c in df.columns if "page name" in c.lower() or c.lower() == "page name"),
        None,
    )
    spend_col = next(
        (c for c in df.columns if "amount spent" in c.lower()),
        None,
    )
    ads_col = next(
        (c for c in df.columns if "number of ads" in c.lower()),
        None,
    )

    if not name_col:
        raise ValueError(f"Cannot find 'Page name' column. Columns: {list(df.columns)}")

    logger.info(
        "Report columns detected — name: '%s', spend: '%s', ads: '%s'",
        name_col, spend_col, ads_col,
    )

    out = pd.DataFrame()
    out["page_name_key"] = df[name_col].apply(normalise_name)

    if spend_col:
        out["adv_total_spend_usd"] = pd.to_numeric(
            df[spend_col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
    else:
        out["adv_total_spend_usd"] = 0.0

    if ads_col:
        out["adv_total_ads"] = pd.to_numeric(
            df[ads_col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
    else:
        out["adv_total_ads"] = 0.0

    # Deduplicate: keep highest-spend row per normalised name
    out = out.sort_values("adv_total_spend_usd", ascending=False).drop_duplicates(
        subset="page_name_key"
    )

    logger.info("Report loaded: %d unique advertisers", len(out))
    return out


def enrich(
    processed_path: Path,
    report_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(processed_path)
    logger.info("Processed ads loaded: %d rows", len(df))

    report = load_report(report_path)

    # Build join key on the ads side
    sponsor_col = next(
        (c for c in df.columns if c.lower() in ("sponsor", "page_name")),
        None,
    )
    if not sponsor_col:
        raise ValueError(
            f"Cannot find sponsor column in processed CSV. Columns: {list(df.columns)}"
        )

    df["_name_key"] = df[sponsor_col].apply(normalise_name)

    df = df.merge(
        report[["page_name_key", "adv_total_spend_usd", "adv_total_ads"]],
        left_on="_name_key",
        right_on="page_name_key",
        how="left",
    ).drop(columns=["_name_key", "page_name_key"], errors="ignore")

    # Fill unmatched advertisers with 0
    df["adv_total_spend_usd"] = df["adv_total_spend_usd"].fillna(0)
    df["adv_total_ads"] = df["adv_total_ads"].fillna(0)

    # Log-transformed versions (used as model features)
    df["adv_log_spend"] = np.log1p(df["adv_total_spend_usd"])
    df["adv_log_ads"] = np.log1p(df["adv_total_ads"])

    matched = (df["adv_total_spend_usd"] > 0).sum()
    logger.info(
        "Join complete: %d / %d ads matched to an advertiser (%.1f%%)",
        matched, len(df), 100 * matched / len(df),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Enriched CSV written to %s", output_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join Meta Ad Library report onto processed ads CSV."
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Path to FacebookAdLibraryReport_*_advertisers.csv",
    )
    parser.add_argument(
        "--processed",
        default="data/processed/political_ads_processed.csv",
        type=Path,
        help="Path to processed ads CSV (output of data_loader)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/political_ads_enriched.csv",
        type=Path,
        help="Where to write the enriched CSV",
    )
    args = parser.parse_args()
    enrich(args.processed, args.report, args.output)


if __name__ == "__main__":
    main()
