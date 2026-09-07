"""MediaWiki API collection for Wikipedia drug articles."""

import re
import time

import mwparserfromhell
import pandas as pd
import requests

MEDIAWIKI_API = "https://{lang}.wikipedia.org/w/api.php"

# Wikipedia policy requires a descriptive User-Agent with contact info; an
# anonymous/generic UA is a common cause of 429 throttling.
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "G6PDDisparityStudy/1.0 "
            "(https://github.com/bgbg/pharma_g6pd; boris@gorelik.net) "
            "research; requests"
        )
    }
)

# Polite throttle floor between API calls (seconds) and retry policy for the
# transient 429/503 responses that otherwise corrupt records as exists=False.
_MIN_INTERVAL = 1.0
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0


def _api_get(lang: str, params: dict) -> dict:
    """GET the MediaWiki API with maxlag + exponential backoff on 429/503.

    Raises on unrecoverable errors so callers can flag a fetch error rather
    than silently recording the article as missing.
    """
    url = MEDIAWIKI_API.format(lang=lang)
    # maxlag tells MediaWiki to shed load gracefully under replication lag.
    params = {**params, "maxlag": 5}
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        resp = _SESSION.get(url, params=params, timeout=30)
        if resp.status_code in (429, 503):
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else _BACKOFF_BASE ** attempt
            print(f"    {resp.status_code} on {lang}; backing off {wait:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait)
            last_exc = requests.HTTPError(f"{resp.status_code} after retries")
            continue
        resp.raise_for_status()
        data = resp.json()
        # maxlag breaches return HTTP 200 with an error envelope; retry those too.
        if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
            time.sleep(_BACKOFF_BASE ** attempt)
            last_exc = requests.HTTPError("maxlag")
            continue
        return data
    raise last_exc or requests.HTTPError("exhausted retries")


def fetch_article(
    lang: str,
    title: str,
    drug: str,
    cpic_risk: str,
    pregnancy_risk: str,
    category: str,
) -> dict:
    """Fetch metadata and wikitext for a single article.

    Returns a dict with all schema fields. If title is empty/None, or if any
    API error occurs, returns a record with exists=False and numeric fields None.
    """
    base = {
        "drug": drug,
        "lang": lang,
        "cpic_risk": cpic_risk,
        "pregnancy_risk": pregnancy_risk,
        "category": category,
        "title": title if title else None,
        # True only when an API error prevented a verdict; distinguishes a
        # genuine missing article (exists=False, fetch_error=False) from a
        # transient failure that must be retried, not treated as "missing".
        "fetch_error": False,
    }

    if not title:
        return {
            **base,
            "exists": False,
            "wikitext": None,
            "length_bytes": None,
            "word_count": None,
            "n_references": None,
            "last_edit_date": None,
            "n_editors_all_time": None,
            "has_infobox": None,
        }

    try:
        # --- metadata call ---
        meta_params = {
            "action": "query",
            "prop": "info|revisions",
            "rvprop": "ids|timestamp|user",
            "rvlimit": "500",
            "titles": title,
            "format": "json",
        }
        meta_json = _api_get(lang, meta_params)
        time.sleep(1)

        pages = meta_json.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))

        # Page missing (e.g. -1 page id)
        if "missing" in page:
            return {
                **base,
                "exists": False,
                "wikitext": None,
                "length_bytes": None,
                "word_count": None,
                "n_references": None,
                "last_edit_date": None,
                "n_editors_all_time": None,
                "has_infobox": None,
            }

        length_bytes = page.get("length")

        revisions = page.get("revisions", [])
        if revisions:
            # Most recent revision is first in the list
            last_edit_date = revisions[0]["timestamp"][:10]  # YYYY-MM-DD
            n_editors_all_time = len({r.get("user", "") for r in revisions})
        else:
            last_edit_date = None
            n_editors_all_time = 0

        # --- wikitext call ---
        parse_params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "format": "json",
        }
        parse_json = _api_get(lang, parse_params)
        time.sleep(1)

        if "error" in parse_json:
            return {
                **base,
                "exists": False,
                "wikitext": None,
                "length_bytes": None,
                "word_count": None,
                "n_references": None,
                "last_edit_date": None,
                "n_editors_all_time": None,
                "has_infobox": None,
            }

        wikitext = parse_json["parse"]["wikitext"]["*"]

        # Strip markup and count words
        parsed = mwparserfromhell.parse(wikitext)
        word_count = len(parsed.strip_code().split())

        # Count references. English/Arabic/Russian editions cite with <ref> tags;
        # the Hebrew edition predominantly uses the inline {{הערה}} ("note") template
        # instead, so count both to avoid undercounting Hebrew citations to zero.
        n_references = wikitext.lower().count("<ref") + len(
            re.findall(r"\{\{הערה[|}]", wikitext)
        )

        # Detect infobox
        has_infobox = "{{infobox" in wikitext.lower()

        return {
            **base,
            "exists": True,
            "wikitext": wikitext,
            "length_bytes": length_bytes,
            "word_count": word_count,
            "n_references": n_references,
            "last_edit_date": last_edit_date,
            "n_editors_all_time": n_editors_all_time,
            "has_infobox": has_infobox,
        }

    except Exception as exc:
        print(f"  ERROR fetching {lang}/{title}: {exc}")
        return {
            **base,
            "fetch_error": True,
            "exists": False,
            "wikitext": None,
            "length_bytes": None,
            "word_count": None,
            "n_references": None,
            "last_edit_date": None,
            "n_editors_all_time": None,
            "has_infobox": None,
        }


def fetch_all(drugs_csv_path) -> list:
    """Load drugs CSV and fetch all drug × language article records.

    Returns a list of dicts, one per drug × language pair.
    """
    df = pd.read_csv(drugs_csv_path)
    langs = ["en", "he", "ar", "ru"]

    records = []
    total = len(df) * len(langs)

    for _, row in df.iterrows():
        drug = row["drug"]
        cpic_risk = row.get("cpic_risk", "")
        pregnancy_risk = row.get("pregnancy_risk", "")
        category = row.get("category", "")
        # Handle NaN from pandas
        if not isinstance(cpic_risk, str):
            cpic_risk = ""
        if not isinstance(pregnancy_risk, str):
            pregnancy_risk = ""
        if not isinstance(category, str):
            category = ""

        for i_lang, lang in enumerate(langs):
            i = len(records) + 1
            title = row.get(f"{lang}_title", "")
            # pandas may read empty cells as NaN (float)
            if not isinstance(title, str) or not title.strip():
                title = ""
            print(f"[{i}/{total}] {drug} / {lang}")
            record = fetch_article(
                lang=lang,
                title=title,
                drug=drug,
                cpic_risk=cpic_risk,
                pregnancy_risk=pregnancy_risk,
                category=category,
            )
            records.append(record)

    return records
