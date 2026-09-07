"""Fetch monthly Wikipedia pageviews for the study articles + two baselines.

All series come from the Wikimedia REST pageviews API with agent=user (bot
traffic already excluded by WMF's classifier):

  data/raw/pageviews.csv          one row per (drug, lang, month)
  data/raw/pageviews_controls.csv one row per (control article, lang, month)
  data/raw/pageviews_totals.csv   one row per (lang, month) — whole-wiki total

The CONTROL COHORT is the baseline that matters. The whole-wiki total is NOT a
fair comparison for a fixed set of articles: it includes the Main Page,
navigation, current events and a constant churn of newly created and trending
pages, so it can hold flat while every established article declines. Comparing
audited drug articles against it manufactures a spurious "drug content is dying
faster" signal that is pure composition effect.

The controls are stable, non-seasonal, non-news encyclopedic articles matched on
the only property that matters here — being a specific reference article people
arrive at from search. If drug articles and controls decline together, the
decline is a property of encyclopedic reference reading, not of medication
information-seeking.

Usage:  python scripts/fetch_pageviews.py [--refresh]
"""

from __future__ import annotations

import sys
import time
from urllib.parse import quote

import pandas as pd
import requests

from g6pd.config import DRUGS_CSV, RAW_DIR

START = "20210701"
END = "20260801"  # exclusive-ish; API returns months whose start is <= END
LANGS = ("en", "he", "ar", "ru")

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
# WMF requires a contact in the UA string for API access.
HEADERS = {"User-Agent": "g6pd-wikipedia-audit/1.0 (boris@gorelik.net)"}

PAGEVIEWS_CSV = RAW_DIR / "pageviews.csv"
CONTROLS_CSV = RAW_DIR / "pageviews_controls.csv"
TOTALS_CSV = RAW_DIR / "pageviews_totals.csv"

# Non-medical baseline cohort, given by English title; the other three language
# titles are resolved via langlinks so the cohort is the same articles everywhere.
CONTROL_ARTICLES = [
    "Photosynthesis",
    "DNA",
    "French Revolution",
    "Plate tectonics",
    "Alexander the Great",
    "Chess",
    "Volcano",
    "Algebra",
    "Coffee",
    "Baroque music",
]


def _get(url: str) -> list[dict]:
    """GET a pageviews endpoint; [] when the API has no data for it (404)."""
    for attempt in range(4):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return []
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2**attempt)
            continue
        r.raise_for_status()
        return r.json()["items"]
    raise RuntimeError(f"gave up on {url}")


def fetch_article(lang: str, title: str) -> list[dict]:
    article = quote(title.replace(" ", "_"), safe="")
    return _get(
        f"{API}/per-article/{lang}.wikipedia/all-access/user/"
        f"{article}/monthly/{START}/{END}"
    )


def fetch_total(lang: str) -> list[dict]:
    return _get(
        f"{API}/aggregate/{lang}.wikipedia/all-access/user/monthly/{START}/{END}"
    )


def control_titles() -> dict[str, dict[str, str]]:
    """{en_title: {lang: title}} for the control cohort, resolved via langlinks."""
    out = {}
    for title in CONTROL_ARTICLES:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title,
                "prop": "langlinks",
                "lllimit": "500",
                "redirects": "1",
                "format": "json",
            },
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        page = next(iter(r.json()["query"]["pages"].values()))
        links = {ll["lang"]: ll["*"] for ll in page.get("langlinks", [])}
        resolved = {"en": title} | {l: links[l] for l in LANGS if l in links}
        # The cohort must be the SAME articles in every language, or its
        # composition shifts between panels and the comparison is worthless.
        if len(resolved) != len(LANGS):
            missing = set(LANGS) - set(resolved)
            raise RuntimeError(f"control {title!r} missing langlinks: {missing}")
        out[title] = resolved
    return out


def main() -> None:
    refresh = "--refresh" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (
        PAGEVIEWS_CSV.exists()
        and CONTROLS_CSV.exists()
        and TOTALS_CSV.exists()
        and not refresh
    ):
        print(f"cached: {PAGEVIEWS_CSV} (--refresh to refetch)")
        return

    drugs = pd.read_csv(DRUGS_CSV)

    rows = []
    for _, drug in drugs.iterrows():
        for lang in LANGS:
            title = drug[f"{lang}_title"]
            if pd.isna(title):
                continue
            items = fetch_article(lang, title)
            if not items:
                print(f"  no data: {lang}:{title}")
            rows.extend(
                {
                    "drug": drug["drug"],
                    "lang": lang,
                    "title": title,
                    "month": it["timestamp"][:6],
                    "views": it["views"],
                }
                for it in items
            )
        print(f"{drug['drug']}: {len(rows)} rows so far")

    pd.DataFrame(rows).to_csv(PAGEVIEWS_CSV, index=False)
    print(f"wrote {PAGEVIEWS_CSV}")

    control_rows = []
    for en_title, titles in control_titles().items():
        for lang, title in titles.items():
            control_rows.extend(
                {
                    "article": en_title,
                    "lang": lang,
                    "title": title,
                    "month": it["timestamp"][:6],
                    "views": it["views"],
                }
                for it in fetch_article(lang, title)
            )
        print(f"control {en_title}: {len(control_rows)} rows so far")
    pd.DataFrame(control_rows).to_csv(CONTROLS_CSV, index=False)
    print(f"wrote {CONTROLS_CSV}")

    totals = [
        {"lang": lang, "month": it["timestamp"][:6], "total_views": it["views"]}
        for lang in LANGS
        for it in fetch_total(lang)
    ]
    pd.DataFrame(totals).to_csv(TOTALS_CSV, index=False)
    print(f"wrote {TOTALS_CSV}")


if __name__ == "__main__":
    main()
