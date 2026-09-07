#!/usr/bin/env python3
"""Audit Wikidata structured data for drug-safety links across three topics.

For each (drug, topic) the audit checks whether the drug's Wikidata item carries
a contraindication (P1909) or significant-drug-interaction (P769) link to the
topic's safety condition. Produces a long-format data/wikidata_gaps.csv and a
summary table.

Topic -> condition QIDs:
    g6pd           -> G6PD deficiency (Q848343)
    teratogenicity -> human pregnancy (Q11995)
    qt             -> long QT syndrome (Q653924), torsades de pointes (Q1625433)

(PGx-SCAR was dropped as an analysed topic 2026-07-01; see methodological_decisions.md.)
"""

import logging
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from g6pd.config import CONTROL_RISK_COLS, DRUGS_CSV, WIKIDATA_GAPS_CSV
from g6pd.wikidata import check_drug_condition_links

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Topic -> (risk column marking a positive, set of condition QIDs)
# PGx-SCAR dropped as an analysed topic 2026-07-01 (see methodological_decisions.md).
TOPICS = {
    "g6pd": ("cpic_risk", {"Q848343"}),  # G6PD deficiency (was wrong QID Q218093; audit #37)
    "teratogenicity": ("pregnancy_risk", {"Q11995"}),
    "qt": ("qt_risk", {"Q653924", "Q1625433"}),
}
# Control detection uses all four original risk columns (incl. pgx_risk) so the
# dropped PGx-only drugs are excluded, not miscounted as negative controls.
RISK_COLS = list(CONTROL_RISK_COLS)


def main() -> None:
    drugs = pd.read_csv(DRUGS_CSV).fillna("")
    for col in RISK_COLS:
        if col not in drugs.columns:
            drugs[col] = ""
    logger.info("Loaded %d drugs from %s", len(drugs), DRUGS_CSV)

    qids = drugs["wikidata_qid"].astype(str).tolist()
    is_control = (drugs[RISK_COLS].apply(lambda s: s.str.strip()) == "").all(axis=1)

    rows = []
    for topic, (risk_col, condition_qids) in TOPICS.items():
        logger.info("Checking %d QIDs for %s links %s ...", len(qids), topic, sorted(condition_qids))
        props_found = check_drug_condition_links(qids, condition_qids)

        for idx, row in drugs.iterrows():
            qid = str(row["wikidata_qid"]).strip()
            found = props_found.get(qid, [])
            has_link = bool(found)
            positive = str(row[risk_col]).strip() != ""

            if is_control[idx]:
                status = "control_unexpected_link" if has_link else "control_no_link_expected"
                if has_link:
                    logger.warning("UNEXPECTED: control %s has %s link to %s via %s",
                                   row["drug"], topic, sorted(condition_qids), found)
            elif positive:
                status = "present_correct" if has_link else "absent"
            else:
                status = "not_applicable"

            rows.append({
                "drug": row["drug"],
                "topic": topic,
                "wikidata_qid": qid,
                "risk_basis": row[risk_col],
                "wikidata_status": status,
                "properties_found": "|".join(found),
            })

    result_df = pd.DataFrame(rows)
    WIKIDATA_GAPS_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(WIKIDATA_GAPS_CSV, index=False)
    logger.info("Wrote %s", WIKIDATA_GAPS_CSV)

    print("\n=== Wikidata audit summary (status x topic) ===")
    pivot = (
        result_df.groupby(["topic", "wikidata_status"]).size().reset_index(name="count")
        .pivot(index="topic", columns="wikidata_status", values="count").fillna(0).astype(int)
    )
    print(pivot.to_string())

    print("\n=== Positives with structured-data coverage (present_correct) ===")
    present = result_df[result_df["wikidata_status"] == "present_correct"]
    if present.empty:
        print("  NONE — no positive drug carries a structured contraindication/interaction "
              "link to its topic condition on Wikidata (across all three topics).")
    else:
        print(present[["drug", "topic", "properties_found"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
