"""
snapshot_drift.py: content drift between the March 2026 article snapshot and the final snapshot (#79).

`data/raw/articles.jsonl` was stored in git on 23/3/2026 (commit 5da0484, first 15 drugs) and again, refreshed
and extended, on 9/6/2026 (commit 7ea8f63, final text; unchanged by the later commit 6278257). The G6PD and
teratogenicity codes of the first 15 drugs were assigned on the March text. For every article that exists in both
versions this script records whether the wikitext changed and by how much the word count moved.

Output (committed): data/snapshot_drift.csv

Usage:
    python scripts/snapshot_drift.py
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.config import DATA_DIR, ROOT_DIR  # noqa: E402

EARLY, FINAL = "5da0484", "6278257"
OUT = DATA_DIR / "snapshot_drift.csv"


def load(commit: str) -> dict:
    txt = subprocess.check_output(
        ["git", "show", f"{commit}:data/raw/articles.jsonl"], cwd=ROOT_DIR, text=True
    )
    return {(r["drug"], r["lang"]): r for r in map(json.loads, txt.splitlines())}


def main():
    early, final = load(EARLY), load(FINAL)
    rows = []
    for (drug, lang), a in sorted(early.items()):
        b = final[(drug, lang)]
        if not (a.get("exists") and b.get("exists")):
            continue
        rows.append({
            "drug": drug, "lang": lang,
            "early_last_edit": a["last_edit_date"], "final_last_edit": b["last_edit_date"],
            "early_words": a["word_count"], "final_words": b["word_count"],
            "text_changed": a["wikitext"] != b["wikitext"],
            "abs_pct_word_change": round(abs(b["word_count"] - a["word_count"]) / a["word_count"] * 100, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ch = df[df["text_changed"]]
    print(f"existing in both: {len(df)}; text changed: {len(ch)}; "
          f"median abs word change {ch['abs_pct_word_change'].median():.2f}%, max {ch['abs_pct_word_change'].max():.1f}%")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
