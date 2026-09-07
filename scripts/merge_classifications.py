#!/usr/bin/env python3
"""Merge LLM classifications into final coded datasets.

Reads llm_classifications.csv, applies human adjudication codes where
consensus was not reached, joins with drug metadata, validates negative
controls, and produces per-topic output files.

Outputs:
    data/coded/g6pd_warnings.csv
    data/coded/teratogenicity_warnings.csv
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Import path constants from config
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.config import (  # noqa: E402
    ARTICLES_JSONL,
    CONTROL_RISK_COLS,
    DRUGS_CSV,
    LLM_CLASSIFICATIONS_CSV,
    TOPIC_WARNINGS,
)


def load_drugs() -> dict:
    """Load drug metadata keyed by drug name."""
    drugs = {}
    with open(DRUGS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drugs[row["drug"]] = row
    return drugs


def load_classifications() -> list[dict]:
    """Load all classifications."""
    with open(LLM_CLASSIFICATIONS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_final_code(row: dict) -> tuple[int, str]:
    """Return (final_code, classification_method) for a row.

    Provenance precedence (audit #41): a cell that a pharmacist adjudicated must
    be tagged ``human_review`` even if its per-model codes were later collapsed
    to a unanimous triple (which flips ``consensus`` to true). Checking the
    adjudication signal *before* the consensus flag keeps the provenance honest;
    otherwise such a cell is silently relabelled ``llm_consensus``.
    """
    human_code = row.get("human_final_code", "")
    adjudicator = row.get("adjudicator", "")
    if human_code != "" and (adjudicator or row["consensus"].lower() != "true"):
        return int(human_code), "human_review"
    if row["consensus"].lower() == "true":
        return int(row["final_code"]), "llm_consensus"
    # Fallback: majority vote (filter out error codes)
    codes = [int(row[f"code_{i}"]) for i in range(1, 4) if int(row[f"code_{i}"]) >= 0]
    if not codes:
        return 0, "majority_vote"
    return Counter(codes).most_common(1)[0][0], "majority_vote"


def model_codes_json(row: dict) -> str:
    """Build JSON string of per-model codes."""
    return json.dumps({
        row["model_1"].split("/")[-1]: int(row["code_1"]),
        row["model_2"].split("/")[-1]: int(row["code_2"]),
        row["model_3"].split("/")[-1]: int(row["code_3"]),
    })


def main():
    drugs = load_drugs()
    classifications = load_classifications()

    # Load article existence from articles.jsonl
    articles_exist = {}
    if ARTICLES_JSONL.exists():
        with open(ARTICLES_JSONL) as f:
            for line in f:
                rec = json.loads(line)
                articles_exist[(rec["drug"], rec["lang"])] = rec.get("exists", False)

    # Group output rows by topic (generalized to all four topics).
    rows_by_topic: dict[str, list[dict]] = {t: [] for t in TOPIC_WARNINGS}
    control_violations = []

    for row in classifications:
        drug_name = row["drug"]
        lang = row["lang"]
        topic = row["topic"]
        if topic not in TOPIC_WARNINGS:
            continue  # ignore any legacy/unknown topic
        drug_meta = drugs.get(drug_name, {})
        risk_col = TOPIC_WARNINGS[topic][1]

        final_code, method = resolve_final_code(row)
        exists = articles_exist.get((drug_name, lang), False)

        out_row = {
            "drug": drug_name,
            "lang": lang,
            "category": drug_meta.get("category", ""),
            risk_col: drug_meta.get(risk_col, ""),
            "exists": str(exists).lower(),
            "warning_code": final_code,
            "warning_location": row["location"],
            "classification_method": method,
            "model_codes": model_codes_json(row),
        }
        rows_by_topic[topic].append(out_row)

        # Negative-control = all four risk columns empty (category may also say "control").
        # Uses CONTROL_RISK_COLS (all four, incl. pgx_risk) so PGx-only drugs are not
        # misread as controls now that PGx-SCAR is no longer in TOPIC_WARNINGS.
        risk_vals = [str(drug_meta.get(c, "")).strip() for c in CONTROL_RISK_COLS]
        is_control = drug_meta.get("category") == "control" or not any(risk_vals)
        if is_control and final_code > 0:
            control_violations.append((drug_name, lang, topic, final_code))

    # Write one CSV per topic that has any rows.
    for topic, (out_csv, risk_col) in TOPIC_WARNINGS.items():
        rows = rows_by_topic[topic]
        if not rows:
            continue
        fields = ["drug", "lang", "category", risk_col, "exists",
                  "warning_code", "warning_location", "classification_method", "model_codes"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        consensus = sum(1 for r in rows if r["classification_method"] == "llm_consensus")
        human = sum(1 for r in rows if r["classification_method"] == "human_review")
        majority = sum(1 for r in rows if r["classification_method"] == "majority_vote")
        print(f"{topic} warnings: {len(rows)} rows -> {out_csv}")
        print(f"  {consensus} consensus, {human} human-reviewed, {majority} majority-vote")

    # Negative control validation
    print()
    if control_violations:
        print("WARNING: NEGATIVE CONTROL VIOLATIONS (faithful readings, not errors, but flag for review):")
        for drug, lang, topic, code in control_violations:
            print(f"  {drug}/{lang} [{topic}] = {code}")
    else:
        print("All negative controls score 0 across all classified topics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
