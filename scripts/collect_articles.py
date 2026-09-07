"""Collect Wikipedia articles for all drug x language pairs and write to JSONL.

Incremental and idempotent: existing good records (exists=True, or genuinely
missing with fetch_error=False) are kept; only pairs that are absent or that
previously hit an API error (fetch_error=True) are (re)fetched. This lets a run
interrupted by 429 throttling be healed by simply re-running, instead of
overwriting good data with "missing" placeholders.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from g6pd import config
from g6pd.fetch import fetch_article

LANGS = ["en", "he", "ar", "ru"]


def load_existing() -> dict:
    """Return {(drug, lang): record} for good (non-errored) existing records."""
    good: dict[tuple[str, str], dict] = {}
    path = config.ARTICLES_JSONL
    if not path.exists():
        return good
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("fetch_error"):
            continue  # force refetch of prior errors
        good[(r["drug"], r["lang"])] = r
    return good


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.DRUGS_CSV).fillna("")
    existing = load_existing()

    records: list[dict] = []
    todo = []
    for _, row in df.iterrows():
        for lang in LANGS:
            key = (row["drug"], lang)
            if key in existing:
                records.append(existing[key])
            else:
                todo.append((row, lang))

    print(f"{len(existing)} good records kept; {len(todo)} to (re)fetch.")
    for i, (row, lang) in enumerate(todo, 1):
        title = row.get(f"{lang}_title", "")
        title = title.strip() if isinstance(title, str) else ""
        print(f"[{i}/{len(todo)}] {row['drug']} / {lang}")
        records.append(
            fetch_article(
                lang=lang,
                title=title,
                drug=row["drug"],
                cpic_risk=row.get("cpic_risk", ""),
                pregnancy_risk=row.get("pregnancy_risk", ""),
                category=row.get("category", ""),
            )
        )

    with config.ARTICLES_JSONL.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_exists = sum(1 for r in records if r.get("exists"))
    n_error = sum(1 for r in records if r.get("fetch_error"))
    n_missing = len(records) - n_exists - n_error
    print(
        f"\nSummary: {len(records)} records | {n_exists} exist | "
        f"{n_missing} genuinely missing | {n_error} fetch_error (re-run to heal)"
    )
    if n_error:
        print("  Re-run this script to retry the errored pairs.")


if __name__ == "__main__":
    main()
