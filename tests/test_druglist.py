"""Regression test: drug-list invariants (audit issue #35).

Locks the structural claims the manuscript makes about data/drugs.csv so a
future edit cannot silently change the study population:

  - 40 drugs total
  - 8 positives in each of the four domains
  - phenytoin is the sole dual-topic drug (teratogen + pharmacogenetic)
  - 9 negative controls, defined as all four risk columns empty
  - the `category` label agrees with the per-topic risk columns
  - Wikidata QIDs are present, well-formed, and unique; every drug has en_title
"""

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DRUGS_CSV = ROOT / "data" / "drugs.csv"

RISK_COLS = ["cpic_risk", "pregnancy_risk", "qt_risk", "pgx_risk"]
CATEGORY_FOR = {
    "cpic_risk": "g6pd_only",
    "pregnancy_risk": "teratogen_only",
    "qt_risk": "qt_only",
    "pgx_risk": "pgx_only",
}
QID_RE = re.compile(r"^Q[1-9][0-9]*$")


@pytest.fixture(scope="module")
def rows():
    with DRUGS_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_total_drug_count(rows):
    assert len(rows) == 40


@pytest.mark.parametrize("col", RISK_COLS)
def test_eight_positives_per_domain(rows, col):
    positives = [r for r in rows if r[col].strip()]
    assert len(positives) == 8


def test_phenytoin_is_sole_dual_topic(rows):
    multi = {r["drug"] for r in rows if sum(1 for c in RISK_COLS if r[c].strip()) > 1}
    assert multi == {"phenytoin"}


def test_phenytoin_domains_are_teratogen_and_pgx(rows):
    phen = next(r for r in rows if r["drug"] == "phenytoin")
    doms = {c for c in RISK_COLS if phen[c].strip()}
    assert doms == {"pregnancy_risk", "pgx_risk"}


def test_nine_controls_all_risk_columns_empty(rows):
    controls = [r for r in rows if not any(r[c].strip() for c in RISK_COLS)]
    assert len(controls) == 9
    assert {r["drug"] for r in controls} == {
        r["drug"] for r in rows if r["category"] == "control"
    }


def test_thirty_one_unique_domain_positive_drugs(rows):
    positive = {r["drug"] for r in rows if any(r[c].strip() for c in RISK_COLS)}
    assert len(positive) == 31  # 40 - 9 controls; phenytoin counted once


def test_category_agrees_with_risk_columns(rows):
    for r in rows:
        doms = [c for c in RISK_COLS if r[c].strip()]
        if len(doms) == 1:
            assert r["category"] == CATEGORY_FOR[doms[0]], r["drug"]
        elif len(doms) == 0:
            assert r["category"] == "control", r["drug"]


def test_wikidata_qids_present_wellformed_unique(rows):
    qids = [r["wikidata_qid"].strip() for r in rows]
    assert all(QID_RE.match(q) for q in qids)
    assert len(set(qids)) == len(qids)


def test_every_drug_has_english_title(rows):
    assert all(r["en_title"].strip() for r in rows)
