"""
Script: code_warnings.py

Reads data/raw/articles.jsonl, applies keyword-based G6PD warning detection
to each article, and writes coded results to data/coded/warnings.csv.

Validates that all negative-control records receive warning_code=0.
Prints a summary table of code counts per language.
"""

import csv
import json
import logging
import sys
from pathlib import Path

# Make the src package importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g6pd.coding import code_article
from g6pd.config import ARTICLES_JSONL, CODED_DIR, WARNINGS_CSV

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "drug",
    "lang",
    "cpic_risk",
    "category",
    "exists",
    "warning_code",
    "warning_location",
    "matched_terms",
]


def main() -> None:
    # ---- Load articles ----
    records = []
    with open(ARTICLES_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), ARTICLES_JSONL)

    # ---- Code each article ----
    rows = []
    for rec in records:
        result = code_article(rec)
        row = {
            "drug": rec["drug"],
            "lang": rec["lang"],
            "cpic_risk": rec.get("cpic_risk", ""),
            "category": rec.get("category", ""),
            "exists": rec.get("exists", False),
            "warning_code": result["warning_code"],
            "warning_location": result["warning_location"],
            "matched_terms": result["matched_terms"],
        }
        rows.append(row)

    # ---- Write CSV ----
    CODED_DIR.mkdir(parents=True, exist_ok=True)
    with open(WARNINGS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), WARNINGS_CSV)

    # ---- Validate negative controls ----
    control_failures = [
        r for r in rows if r["category"] == "control" and r["warning_code"] != 0
    ]
    if control_failures:
        for r in control_failures:
            logger.warning(
                "Negative control scored > 0: drug=%s lang=%s code=%s location=%s matched=%s",
                r["drug"], r["lang"], r["warning_code"],
                r["warning_location"], r["matched_terms"],
            )
    else:
        logger.info("All negative-control records correctly scored 0.")

    # ---- Summary table ----
    langs = ["en", "he", "ar", "ru"]
    codes = [0, 1, 2]

    # Build counts
    counts: dict[str, dict[int, int]] = {lang: {c: 0 for c in codes} for lang in langs}
    for r in rows:
        lang = r["lang"]
        code = int(r["warning_code"])
        if lang in counts and code in counts[lang]:
            counts[lang][code] += 1

    # Print table
    print()
    print(f"{'lang':<6}| {'code_0':>6} | {'code_1':>6} | {'code_2':>6}")
    print("-" * 31)
    for lang in langs:
        c = counts[lang]
        print(f"{lang:<6}| {c[0]:>6} | {c[1]:>6} | {c[2]:>6}")
    print()

    # ---- Quick acceptance checks ----
    en_code2 = [r for r in rows if r["lang"] == "en" and int(r["warning_code"]) == 2]
    if en_code2:
        logger.info(
            "Acceptance check PASSED: %d EN CPIC drug(s) scored 2: %s",
            len(en_code2),
            [r["drug"] for r in en_code2],
        )
    else:
        logger.warning(
            "Acceptance check FAILED: no EN CPIC drug received warning_code=2"
        )

    total_rows = len(rows)
    if total_rows == 40:
        logger.info("Row count check PASSED: %d rows", total_rows)
    else:
        logger.warning("Row count check: expected 40 rows, got %d", total_rows)


if __name__ == "__main__":
    main()
