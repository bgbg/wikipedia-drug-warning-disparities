"""
assemble_four_topic_coded.py — build the combined four-topic coded dataset (M3, #31).

Option B (see methodological_decisions.md, 2026-06-09): the existing G6PD and
teratogenicity codes are kept exactly as published (the adjudicated 25-drug
finals in paper/data/), and the QT topic carries the June-2026 snapshot. This
script concatenates the topics into the long format the ordinal model reads
(drug, lang, topic, final_code, human_final_code), so the hierarchical fit sees
every (drug, language, topic) cell.

PGx-SCAR was dropped 2026-07-01 (see methodological_decisions.md); the output
filename keeps its historical "four_topic" name but now covers three topics.

Sources:
    paper/data/g6pd_warnings.csv            (topic = g6pd, column warning_code)
    paper/data/teratogenicity_warnings.csv  (topic = teratogenicity)
    data/coded/qt_warnings.csv              (topic = qt; from merge_classifications)

Output:
    data/coded/coded_four_topic.csv

Usage:
    python scripts/assemble_four_topic_coded.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.config import CODED_DIR, QT_WARNINGS_CSV, ROOT_DIR  # noqa: E402

PAPER_DATA = ROOT_DIR / "paper" / "data"

# PGx-SCAR dropped 2026-07-01 (Maya + team verdict). The output filename keeps its
# historical "four_topic" name to avoid breaking downstream references; it now
# assembles three topics. Re-run this script to regenerate the pgx-free dataset.
SOURCES = {
    "g6pd": PAPER_DATA / "g6pd_warnings.csv",
    "teratogenicity": PAPER_DATA / "teratogenicity_warnings.csv",
    "qt": QT_WARNINGS_CSV,
}
OUT = CODED_DIR / "coded_four_topic.csv"


def main() -> None:
    frames = []
    for topic, path in SOURCES.items():
        if not path.exists():
            print(f"WARNING: missing {path} for topic {topic}; skipping", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        out = pd.DataFrame({
            "drug": df["drug"],
            "lang": df["lang"],
            "topic": topic,
            "final_code": df["warning_code"].astype(int),
            "human_final_code": "",  # adjudicated codes already folded into warning_code
        })
        frames.append(out)
        print(f"{topic:15s} {len(out):3d} rows from {path.name}")

    combined = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False)
    print(f"\nWrote {len(combined)} rows ({combined['topic'].nunique()} topics) -> {OUT}")
    print("Code distribution by topic:")
    print(combined.pivot_table(index="topic", columns="final_code", aggfunc="size", fill_value=0))


if __name__ == "__main__":
    main()
