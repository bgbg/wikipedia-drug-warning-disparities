"""
fetch_revision_ids.py: recover the archived Wikipedia revision IDs behind the stored article snapshot (#79).

`data/raw/articles.jsonl` stores the wikitext and a `last_edit_date` (date only) but neither a revision ID
nor a retrieval time. For every existing article of the 33-drug panel this script asks the MediaWiki API for
the revisions made on `last_edit_date` (UTC) and compares each revision's wikitext with the stored wikitext:

    exact_match   a revision on that date has wikitext identical to the stored text (ID established)
    date_only     no text match; the latest revision on that date is listed (ID not established)
    ambiguous     several revisions share the date and none matches the stored text
    not_found     no revision on that date (title moved, or date not in the revision history)

Only `exact_match` rows carry an ID the manuscript may cite as archived.

Output (committed): data/snapshot_revisions.csv

Usage:
    python scripts/fetch_revision_ids.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.config import ARTICLES_JSONL, DATA_DIR, DRUGS_CSV  # noqa: E402
from g6pd.fetch import _api_get  # noqa: E402

OUT = DATA_DIR / "snapshot_revisions.csv"
RISK_COLS = ["cpic_risk", "pregnancy_risk", "qt_risk"]
PAUSE = 1.0


def panel_drugs() -> set[str]:
    """The 33 analysed drugs: positives for any analysed topic plus the clean controls."""
    d = pd.read_csv(DRUGS_CSV).fillna("")
    positive = d[RISK_COLS].astype(str).apply(lambda r: any(v.strip() for v in r), axis=1)
    control = (d[RISK_COLS + ["pgx_risk"]].astype(str).apply(lambda r: not any(v.strip() for v in r), axis=1))
    return set(d.loc[positive | control, "drug"])


def sha(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode()).hexdigest()


def revisions_on_date(lang: str, title: str, date: str) -> tuple[list[dict], str]:
    """Revisions on `date` (UTC), newest first, with wikitext. Returns (revisions, resolved_title)."""
    data = _api_get(lang, {
        "action": "query", "format": "json", "formatversion": 2, "titles": title, "redirects": 1,
        "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main",
        "rvlimit": 50, "rvdir": "older",
        "rvstart": f"{date}T23:59:59Z", "rvend": f"{date}T00:00:00Z",
    })
    page = data["query"]["pages"][0]
    revs = page.get("revisions", []) if not page.get("missing") else []
    out = [{
        "revid": r["revid"], "timestamp": r["timestamp"],
        "sha": sha(r.get("slots", {}).get("main", {}).get("content", "")),
    } for r in revs]
    return out, page.get("title", title)


def classify(revs: list[dict], stored_sha: str) -> dict:
    if not revs:
        return {"status": "not_found", "revision_id": "", "revision_timestamp": ""}
    hit = next((r for r in revs if r["sha"] == stored_sha), None)
    if hit:
        return {"status": "exact_match", "revision_id": hit["revid"], "revision_timestamp": hit["timestamp"]}
    status = "ambiguous" if len(revs) > 1 else "date_only"
    return {"status": status, "revision_id": "", "revision_timestamp": ""}


def main():
    panel = panel_drugs()
    rows = []
    for line in open(ARTICLES_JSONL):
        r = json.loads(line)
        if not (r["drug"] in panel and r.get("exists")):
            continue
        date = r["last_edit_date"]
        # an API failure propagates (after the retries in `_api_get`): no partial file, no guessed ID
        revs, resolved = revisions_on_date(r["lang"], r["title"], date)
        res = classify(revs, sha(r["wikitext"]))
        n = len(revs)
        rows.append({"drug": r["drug"], "lang": r["lang"], "title": resolved, "last_edit_date": date,
                     "n_revisions_on_date": n, **res})
        print(f"{r['lang']} {r['drug']}: {res['status']} ({n} on {date})", flush=True)
        time.sleep(PAUSE)
    df = pd.DataFrame(rows).sort_values(["lang", "drug"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print("\n", df["status"].value_counts().to_string())
    print(f"articles: {len(df)}; last_edit_date range {df['last_edit_date'].min()} to {df['last_edit_date'].max()}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
