"""Regression test for the Wikidata structured-data audit (issue #37).

Locks the headline 0/24 invariant against the *stored* data/wikidata_gaps.csv.
(PGx-SCAR dropped as an analysed topic 2026-07-01; three domains x 8 = 24.)
No network: the test reads only the frozen CSV, so it never depends on live
Wikidata. The live 18/40 P769 sanity figure is re-derived separately by
audit/rederive/rederive_wikidata_37.py --live and is intentionally not asserted
here (Wikidata is mutable structured data).
"""

import csv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GAPS_CSV = REPO / "data" / "wikidata_gaps.csv"
POSITIVE_STATUSES = {"absent", "present_correct"}
EXPECTED_TOPICS = {"g6pd", "teratogenicity", "qt"}


@pytest.fixture(scope="module")
def rows():
    with GAPS_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_gaps_csv_exists():
    assert GAPS_CSV.is_file(), f"missing {GAPS_CSV}"


def test_twentyfour_domain_positive_pairs(rows):
    positives = [r for r in rows if r["wikidata_status"] in POSITIVE_STATUSES]
    assert len(positives) == 24


def test_eight_positives_per_topic(rows):
    by_topic = {}
    for r in rows:
        if r["wikidata_status"] in POSITIVE_STATUSES:
            by_topic[r["topic"]] = by_topic.get(r["topic"], 0) + 1
    assert set(by_topic) == EXPECTED_TOPICS
    assert all(count == 8 for count in by_topic.values()), by_topic


def test_zero_of_twentyfour_carry_a_link(rows):
    """The headline invariant: no domain-positive drug carries a P1909/P769
    link to its safety condition."""
    present_correct = [r for r in rows if r["wikidata_status"] == "present_correct"]
    assert present_correct == []


def test_no_properties_found_anywhere(rows):
    """No row records any matching structured property at all."""
    nonempty = [r for r in rows if r["properties_found"].strip()]
    assert nonempty == []
