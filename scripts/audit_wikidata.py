#!/usr/bin/env python3
"""Audit Wikidata structured data for drug-safety links across three topics.

For each panel drug (33 drugs: the 40 in data/drugs.csv minus the 7 PGx-only
drugs dropped 2026-07-01) and each topic, the audit checks the drug's Wikidata
item for the properties below, then compares the positives with two base rates.

Topic -> primary property (counted) -> target:
    g6pd           P10946 "contraindicated in case of" -> G6PD deficiency (Q848343)
    qt             P10946                              -> long QT (Q653924), torsades (Q1625433)
    teratogenicity P3489  "pregnancy category"         -> any value on the drug
                   P10946                              -> pregnancy (Q11995)
Context only (reported, never counted): P1909 "side effect" and P769
"significant drug interaction" (which links two drugs), same targets.
Labels are re-fetched with wbgetentities into data/wikidata_properties.json.
(Earlier versions wrongly called P1909 a contraindication; issue #76.)

Denominator rule (PENDING MAYA CONFIRMATION): positives whose reference standard
says "contraindicated" (G6PD, teratogenicity); QT is reported separately.

Base rates: (a) the 33 panel drugs, (b) all Wikidata items with an ATC code
(P267), counted with SPARQL COUNT queries (fast, no paging needed), and (c) raw
p: statement counts per property on all of Wikidata (unit: statements, not items).

Outputs: data/wikidata_properties.json, data/wikidata_gaps.csv (long format with
a `property` column), data/wikidata_base_rates.csv.
"""

import datetime
import json
import logging
import sys
import time

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from g6pd.config import (CONTROL_RISK_COLS, DRUGS_CSV, WIKIDATA_BASE_RATES_CSV,
                         WIKIDATA_GAPS_CSV, WIKIDATA_PROPERTIES_JSON)
from g6pd.wikidata import (ATC_PROP, PROPERTY_IDS, count_atc_items, count_statements,
                           fetch_drug_claims, fetch_labels, in_denominator,
                           matching_values, topic_checks)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TOPIC_RISK_COL = {"g6pd": "cpic_risk", "teratogenicity": "pregnancy_risk", "qt": "qt_risk"}
TODAY = datetime.date.today().isoformat()
POLITE_PAUSE_S = 1.0  # between WDQS queries


def load_panel() -> pd.DataFrame:
    drugs = pd.read_csv(DRUGS_CSV).fillna("")
    topic_cols = list(TOPIC_RISK_COL.values())
    # PGx-only drugs (pgx_risk set, no analysed-topic risk) are excluded, not controls.
    panel = drugs[~((drugs[topic_cols].apply(lambda s: s.str.strip()) == "").all(axis=1)
                    & (drugs["pgx_risk"].str.strip() != ""))].reset_index(drop=True)
    panel["is_control"] = (panel[list(CONTROL_RISK_COLS)].apply(lambda s: s.str.strip()) == "").all(axis=1)
    if (panel["wikidata_qid"].str.strip() == "").any():
        raise SystemExit("panel drug without a wikidata_qid")
    return panel


def save_properties() -> None:
    labels = fetch_labels(list(PROPERTY_IDS))
    payload = {"retrieved_on": TODAY, "source": "wbgetentities", "language": "en",
               "properties": {p: labels[p] for p in PROPERTY_IDS}}
    WIKIDATA_PROPERTIES_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for p in PROPERTY_IDS:
        logger.info("%s = %r", p, labels[p]["label"])


def build_gaps(panel: pd.DataFrame, claims: dict, value_labels: dict) -> pd.DataFrame:
    rows = []
    for topic, risk_col in TOPIC_RISK_COL.items():
        for prop, role, targets in topic_checks(topic):
            for _, d in panel.iterrows():
                qid = d["wikidata_qid"].strip()
                found = matching_values(claims.get(qid, {}).get(prop, []), targets)
                basis = str(d[risk_col]).strip()
                positive = basis != ""
                if role == "primary" and prop != "P3489" and d["is_control"] and found:
                    logger.warning("UNEXPECTED: control %s has %s link %s", d["drug"], prop, found)
                rows.append({
                    "drug": d["drug"], "topic": topic, "wikidata_qid": qid, "risk_basis": basis,
                    "property": prop, "role": role, "is_control": bool(d["is_control"]),
                    "in_denominator": positive and in_denominator(topic, basis),
                    "has_link": bool(found),
                    "linked_values": "|".join(found),
                    "linked_labels": "|".join(value_labels.get(v, {}).get("label", v) for v in found),
                    "wikidata_status": ("present_correct" if found else "absent") if positive else "not_applicable",
                })
    return pd.DataFrame(rows)


def rate_row(scope, topic, prop, role, targets, n_total, n_with, note=""):
    return {"scope": scope, "topic": topic, "property": prop, "role": role,
            "targets": "|".join(sorted(targets)) if targets else "any",
            "n_total": "" if n_total is None else n_total, "n_with_link": n_with,
            "proportion": round(n_with / n_total, 4) if n_total else "",
            "measured_on": TODAY, "note": note}


def build_base_rates(gaps: pd.DataFrame, panel: pd.DataFrame, claims: dict) -> pd.DataFrame:
    rows = []
    # (1) Positives: denominator rule (G6PD, teratogenicity) and QT reported separately.
    for (topic, prop, role), g in gaps.groupby(["topic", "property", "role"], sort=False):
        pos = g[g["wikidata_status"] != "not_applicable"]
        den = pos[pos["in_denominator"]]
        targets = next(t for p, r, t in topic_checks(topic) if p == prop and r == role)
        if topic == "qt":
            rows.append(rate_row("qt_positives_separate", topic, prop, role, targets,
                                 len(pos), int(pos["has_link"].sum()), "Known Risk of TdP, not a contraindication"))
        else:
            rows.append(rate_row("denominator_contraindicated", topic, prop, role, targets,
                                 len(den), int(den["has_link"].sum()), "rule pending Maya confirmation"))
    # (2) Base rates: all 33 panel drugs, and all Wikidata items with an ATC code.
    keys = [(t, p, r, tg) for t in TOPIC_RISK_COL for p, r, tg in topic_checks(t)]
    keys += [("any", p, "property_overall", None) for p in (*PROPERTY_IDS, ATC_PROP)]
    n_atc = count_atc_items()
    for topic, prop, role, targets in keys:
        n_panel = sum(bool(matching_values(claims.get(q, {}).get(prop, []), targets))
                      for q in panel["wikidata_qid"].str.strip())
        rows.append(rate_row("panel_33", topic, prop, role, targets, len(panel), n_panel))
        time.sleep(POLITE_PAUSE_S)
        rows.append(rate_row("wikidata_atc_items", topic, prop, role, targets, n_atc,
                             count_atc_items(prop, targets), "items with P267, truthy statements"))
    # (3) Raw statement counts per property on all of Wikidata (unit: p: statements,
    # not distinct items; n_total is blank because there is no denominator).
    for prop in PROPERTY_IDS:
        time.sleep(POLITE_PAUSE_S)
        rows.append(rate_row("wikidata_statements", "any", prop, "property_overall", None, None,
                             count_statements(prop), "unit: p: statements on all items, not distinct items"))
    return pd.DataFrame(rows)


def main() -> None:
    panel = load_panel()
    logger.info("Panel: %d drugs", len(panel))
    save_properties()

    props = (*PROPERTY_IDS, ATC_PROP)
    claims = fetch_drug_claims(panel["wikidata_qid"].str.strip().tolist(), props)
    all_vals = sorted({v for c in claims.values() for p in PROPERTY_IDS for v in c.get(p, [])
                       if v.startswith("Q")})
    value_labels = fetch_labels(all_vals) if all_vals else {}

    gaps = build_gaps(panel, claims, value_labels)
    gaps.to_csv(WIKIDATA_GAPS_CSV, index=False)
    logger.info("Wrote %s (%d rows)", WIKIDATA_GAPS_CSV, len(gaps))

    rates = build_base_rates(gaps, panel, claims)
    rates.to_csv(WIKIDATA_BASE_RATES_CSV, index=False)
    logger.info("Wrote %s (%d rows)", WIKIDATA_BASE_RATES_CSV, len(rates))

    print("\n=== Base rates (n_with_link / n_total) ===")
    print(rates[["scope", "topic", "property", "role", "targets", "n_with_link", "n_total"]].to_string(index=False))
    print("\n=== Primary-property links on positives (present_correct) ===")
    hit = gaps[(gaps["wikidata_status"] == "present_correct") & (gaps["role"] == "primary")]
    print(hit[["drug", "topic", "property", "linked_labels"]].to_string(index=False) if len(hit) else "  NONE")


if __name__ == "__main__":
    main()
