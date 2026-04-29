"""
Meta Ad Library API scraper for U.S. political ads.

Handles:
- Cursor-based pagination
- Exponential backoff on rate limits (error 613)
- Incremental JSONL output with deduplication so runs are resumable
- Access token read from environment only (never hardcoded)

Required permissions on the token:
  - ads_read
  - The account must have been granted Ad Library API access at
    https://www.facebook.com/ads/library/api/
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Sequence

import requests

logger = logging.getLogger(__name__)

# Stable Graph API version; update if Meta deprecates it
_API_VERSION = "v21.0"
_BASE_URL = f"https://graph.facebook.com/{_API_VERSION}/ads_archive"

# Fields returned per ad.  impressions/spend come back as
# {"lower_bound": "1000", "upper_bound": "4999"} which the existing
# data_loader._extract_numeric_bounds() already handles.
_FIELDS = [
    "id",
    "ad_creation_time",
    "ad_creative_bodies",
    "ad_creative_link_titles",
    "ad_delivery_start_time",
    "ad_delivery_stop_time",
    "page_id",
    "page_name",
    "bylines",
    "currency",
    "impressions",
    "spend",
    "delivery_by_region",
    "demographic_distribution",
    "publisher_platforms",
    "languages",
    "ad_snapshot_url",
]


class MetaAdLibraryScraper:
    """Scrapes U.S. political ads from the Meta Ad Library API."""

    def __init__(
        self,
        access_token: str,
        output_path: str | Path,
        date_min: str = "2022-01-01",
        date_max: str = "2023-12-31",
        country: str = "US",
        ad_type: str = "POLITICAL_AND_ISSUE_ADS",
        limit: int = 100,
        max_retries: int = 6,
        base_retry_delay: float = 60.0,
    ) -> None:
        if not access_token:
            raise ValueError(
                "access_token is empty. Set META_ACCESS_TOKEN in your .env file."
            )
        # Store the token only in memory; never log or write it
        self._token = access_token
        self.output_path = Path(output_path)
        self.date_min = date_min
        self.date_max = date_max
        self.country = country
        self.ad_type = ad_type
        self.limit = min(limit, 100)  # API hard cap is 100
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self._session = requests.Session()
        self._seen_ids: set[str] = self._load_seen_ids()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_seen_ids(self) -> set[str]:
        """Read ad IDs already saved to the JSONL file to allow resuming."""
        seen: set[str] = set()
        if not self.output_path.exists():
            return seen
        with open(self.output_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ad_id = str(obj.get("id", ""))
                    if ad_id:
                        seen.add(ad_id)
                except json.JSONDecodeError:
                    pass
        if seen:
            logger.info(
                "Resuming: loaded %d existing ad IDs from %s",
                len(seen),
                self.output_path,
            )
        return seen

    def _get(self, params: dict) -> dict:
        """One GET to the ads_archive endpoint with retry/back-off."""
        # Never log the full params dict — it contains the access token
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(_BASE_URL, params=params, timeout=30)
                payload = resp.json()
            except requests.RequestException as exc:
                wait = 2**attempt * 10
                logger.warning(
                    "Network error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue

            if "error" in payload:
                err = payload["error"]
                code = err.get("code", 0)
                msg = err.get("message", "")
                if code == 613:
                    # Rate-limit: back off with exponential delay
                    wait = 2**attempt * self.base_retry_delay
                    logger.warning(
                        "Rate limited (attempt %d/%d) — waiting %.0fs",
                        attempt + 1,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                if code == 190:
                    raise RuntimeError(
                        "Invalid or expired access token (error 190). "
                        "Refresh META_ACCESS_TOKEN in your .env file."
                    )
                raise RuntimeError(f"API error {code}: {msg}")

            return payload

        raise RuntimeError(
            f"API request failed after {self.max_retries} attempts."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scrape_term(self, search_term: str) -> int:
        """
        Fetch all paginated results for *search_term* and append new ads to
        the JSONL file.  Returns the count of newly saved ads.
        """
        params: dict = {
            "search_terms": search_term,
            "ad_type": self.ad_type,
            "ad_reached_countries": f"['{self.country}']",
            "ad_delivery_date_min": self.date_min,
            "ad_delivery_date_max": self.date_max,
            "fields": ",".join(_FIELDS),
            "limit": self.limit,
            "access_token": self._token,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        new_count = 0
        page_num = 0

        with open(self.output_path, "a", encoding="utf-8") as fh:
            while True:
                payload = self._get(params)
                ads = payload.get("data", [])

                if not ads:
                    break

                for ad in ads:
                    ad_id = str(ad.get("id", ""))
                    if ad_id and ad_id not in self._seen_ids:
                        self._seen_ids.add(ad_id)
                        fh.write(json.dumps(ad, ensure_ascii=False) + "\n")
                        new_count += 1

                page_num += 1
                logger.info(
                    "  term='%s' page=%d ads_on_page=%d new_total=%d",
                    search_term,
                    page_num,
                    len(ads),
                    new_count,
                )

                paging = payload.get("paging", {})
                after_cursor = paging.get("cursors", {}).get("after")
                if not after_cursor or "next" not in paging:
                    break

                params["after"] = after_cursor
                time.sleep(0.5)  # polite inter-page delay

        return new_count

    def scrape_all(self, search_terms: Sequence[str]) -> int:
        """
        Iterate over every term in *search_terms*, deduplicating across terms.
        Returns total count of newly saved ads.
        """
        total = 0
        for i, term in enumerate(search_terms, 1):
            logger.info(
                "[%d/%d] Scraping term: '%s'  (seen so far: %d unique ads)",
                i,
                len(search_terms),
                term,
                len(self._seen_ids),
            )
            n = self.scrape_term(term)
            total += n
            logger.info("  → %d new ads (running total: %d)", n, total)
            time.sleep(2)  # polite inter-term delay

        logger.info("Scrape complete. %d new ads saved to %s", total, self.output_path)
        return total
