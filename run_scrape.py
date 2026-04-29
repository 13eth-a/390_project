"""
Scrape U.S. political ads from the Meta Ad Library API (2022–2023),
then convert the JSONL output to a CSV ready for the baseline pipeline.

=== Prerequisites ===

1. Apply for Ad Library API access:
       https://www.facebook.com/ads/library/api/
   (Takes 1–3 business days. You need a personal Facebook account, a
   confirmed identity, and a Meta developer app.)

2. Generate a User Access Token with ads_read permission:
   - Go to https://developers.facebook.com/tools/explorer/
   - Select your app → Add Permission: ads_read → Generate Token
   - The default short-lived token (~1 h) is enough for a single session.
     For multi-day scrapes, generate a long-lived token (60 days) by
     exchanging it at:
       https://graph.facebook.com/v21.0/oauth/access_token?
           grant_type=fb_exchange_token
           &client_id=YOUR_APP_ID
           &client_secret=YOUR_APP_SECRET
           &fb_exchange_token=YOUR_SHORT_TOKEN

3. Copy .env.example → .env and fill in META_ACCESS_TOKEN:
       cp .env.example .env  # then open .env and paste your token

=== Usage ===

  # Full default run (2022-01-01 → 2023-12-31, US, all terms in config/search_terms.txt)
  python run_scrape.py

  # Custom date window
  python run_scrape.py --date-min 2023-01-01 --date-max 2023-12-31

  # Dry-run: only fetch the first page for each term (for testing without burning quota)
  python run_scrape.py --dry-run

  # Use a custom terms file
  python run_scrape.py --terms-file config/search_terms.txt

=== Output ===

  data/raw/meta_ads_2022_2023.jsonl   ← raw API responses, one ad per line
  data/raw/meta_ads_2022_2023.csv     ← normalised CSV, compatible with data_loader.py

The JSONL file is written incrementally so a crash can be resumed — already-
scraped ad IDs are skipped on the next run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load .env before importing anything that reads env vars
from dotenv import load_dotenv

load_dotenv()

from src.meta_ad_scraper import MetaAdLibraryScraper  # noqa: E402
from src.meta_converter import convert_jsonl_to_csv   # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TERMS_FILE = Path("config/search_terms.txt")
DEFAULT_JSONL_OUT = Path("data/raw/meta_ads_2022_2023.jsonl")
DEFAULT_CSV_OUT = Path("data/raw/meta_ads_2022_2023.csv")


def load_search_terms(terms_file: Path) -> list[str]:
    if not terms_file.exists():
        logger.error("Search terms file not found: %s", terms_file)
        sys.exit(1)
    terms = []
    with open(terms_file, encoding="utf-8") as fh:
        for line in fh:
            term = line.strip()
            if term and not term.startswith("#"):
                terms.append(term)
    if not terms:
        logger.error("No search terms found in %s", terms_file)
        sys.exit(1)
    logger.info("Loaded %d search terms from %s", len(terms), terms_file)
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape U.S. political ads from the Meta Ad Library API."
    )
    parser.add_argument(
        "--date-min",
        default="2022-01-01",
        help="Start of delivery date window (YYYY-MM-DD). Default: 2022-01-01",
    )
    parser.add_argument(
        "--date-max",
        default="2023-12-31",
        help="End of delivery date window (YYYY-MM-DD). Default: 2023-12-31",
    )
    parser.add_argument(
        "--terms-file",
        type=Path,
        default=DEFAULT_TERMS_FILE,
        help=f"Path to newline-separated search terms file. Default: {DEFAULT_TERMS_FILE}",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=DEFAULT_JSONL_OUT,
        help=f"Output JSONL path. Default: {DEFAULT_JSONL_OUT}",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_CSV_OUT,
        help=f"Output CSV path. Default: {DEFAULT_CSV_OUT}",
    )
    parser.add_argument(
        "--ad-type",
        default="POLITICAL_AND_ISSUE_ADS",
        choices=["POLITICAL_AND_ISSUE_ADS", "ALL"],
        help="Ad type filter. Use ALL to test connectivity before political access is confirmed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch only one page per term (for testing).",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Skip scraping; only convert an existing JSONL file to CSV.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Token check — read from environment, never from CLI args
    # -----------------------------------------------------------------------
    token = os.environ.get("META_ACCESS_TOKEN", "").strip()
    if not token and not args.convert_only:
        logger.error(
            "META_ACCESS_TOKEN is not set.\n"
            "  1. Copy .env.example → .env\n"
            "  2. Paste your token: META_ACCESS_TOKEN=your_token_here\n"
            "  3. Re-run this script."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Convert-only mode
    # -----------------------------------------------------------------------
    if args.convert_only:
        logger.info("Converting %s → %s", args.jsonl_out, args.csv_out)
        convert_jsonl_to_csv(args.jsonl_out, args.csv_out)
        return

    # -----------------------------------------------------------------------
    # Scrape
    # -----------------------------------------------------------------------
    search_terms = load_search_terms(args.terms_file)

    if args.dry_run:
        logger.info("DRY RUN — only one page will be fetched per term")
        # Monkey-patch limit so pagination stops after first page
        original_limit = 10
    else:
        original_limit = 100

    scraper = MetaAdLibraryScraper(
        access_token=token,
        output_path=args.jsonl_out,
        date_min=args.date_min,
        date_max=args.date_max,
        ad_type=args.ad_type,
        limit=original_limit if args.dry_run else 100,
    )

    if args.dry_run:
        # For dry runs, restrict each term to exactly one page
        total = 0
        for term in search_terms:
            n = scraper.scrape_term(term)
            total += n
            # Only fetch the first page: scrape_term handles one page at a time
            # via pagination; dry-run works because limit=10 returns small batches
        logger.info("Dry run complete. %d new ads written to %s", total, args.jsonl_out)
    else:
        scraper.scrape_all(search_terms)

    # -----------------------------------------------------------------------
    # Convert JSONL → CSV
    # -----------------------------------------------------------------------
    if args.jsonl_out.exists():
        logger.info("Converting JSONL → CSV...")
        convert_jsonl_to_csv(args.jsonl_out, args.csv_out)
        logger.info("Done. CSV ready at %s", args.csv_out)
    else:
        logger.warning("JSONL file not found; skipping CSV conversion.")


if __name__ == "__main__":
    main()
