"""Regression test for the Wikidata structured-data audit (issues #37, #76).

Checks the *stored* data/wikidata_gaps.csv, data/wikidata_base_rates.csv and
data/wikidata_properties.json for internal consistency. No network: the tests
read frozen files only, so they never depend on live Wikidata (which is mutable).

Issue #76 replaced the original audit, which treated P1909 ("side effect") and
P769 (drug-drug interaction) as contraindication links, with the corrected
topic-to-property mapping (P10946 and P3489 primary; P1909 and P769 context).
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from g6pd.wikidata import PROPERTY_IDS, topic_checks  # noqa: E402

GAPS_CSV = REPO / "data" / "wikidata_gaps.csv"
RATES_CSV = REPO / "data" / "wikidata_base_rates.csv"
PROPS_JSON = REPO / "data" / "wikidata_properties.json"
TOPICS = ("g6pd", "teratogenicity", "qt")


def _read(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def gaps():
    return _read(GAPS_CSV)


@pytest.fixture(scope="module")
def rates():
    return _read(RATES_CSV)


def test_gaps_has_property_column_and_33_drugs(gaps):
    assert "property" in gaps[0]
    assert len({r["drug"] for r in gaps}) == 33  # 40 minus the 7 PGx-only drugs


def test_checks_match_topic_mapping(gaps):
    for topic in TOPICS:
        expected = {(p, r) for p, r, _ in topic_checks(topic)}
        assert {(r["property"], r["role"]) for r in gaps if r["topic"] == topic} == expected


def test_p1909_is_never_a_primary_property(gaps):
    assert not [r for r in gaps if r["property"] == "P1909" and r["role"] == "primary"]


def test_eight_positives_per_topic(gaps):
    for topic in TOPICS:
        pos = {r["drug"] for r in gaps if r["topic"] == topic and r["wikidata_status"] != "not_applicable"}
        assert len(pos) == 8, topic


def test_denominator_is_five_g6pd_plus_eight_teratogens(gaps):
    den = {(r["topic"], r["drug"]) for r in gaps if r["in_denominator"] == "True"}
    assert sum(t == "g6pd" for t, _ in den) == 5
    assert sum(t == "teratogenicity" for t, _ in den) == 8
    assert not [t for t, _ in den if t == "qt"]


def test_denominator_rows_are_a_subset_of_positives(gaps):
    assert not [r for r in gaps if r["in_denominator"] == "True" and r["wikidata_status"] == "not_applicable"]


def test_base_rates_agree_with_gaps(rates, gaps):
    """Denominator and QT rows of the base-rate table are re-derivable from the gaps file."""
    for r in rates:
        if r["scope"] not in ("denominator_contraindicated", "qt_positives_separate"):
            continue
        rows = [g for g in gaps if g["topic"] == r["topic"] and g["property"] == r["property"]
                and g["role"] == r["role"] and g["wikidata_status"] != "not_applicable"
                and (r["scope"] == "qt_positives_separate" or g["in_denominator"] == "True")]
        assert int(r["n_total"]) == len(rows)
        assert int(r["n_with_link"]) == sum(g["has_link"] == "True" for g in rows)


def test_panel_and_atc_scopes_present(rates):
    scopes = {r["scope"] for r in rates}
    assert {"denominator_contraindicated", "qt_positives_separate", "panel_33", "wikidata_atc_items"} <= scopes
    assert {int(r["n_total"]) for r in rates if r["scope"] == "panel_33"} == {33}
    assert all(int(r["n_total"]) > 1000 for r in rates if r["scope"] == "wikidata_atc_items")


def test_property_labels_saved():
    data = json.loads(PROPS_JSON.read_text())["properties"]
    assert set(data) == set(PROPERTY_IDS)
    assert data["P1909"]["label"] == "side effect"
    assert data["P769"]["label"] == "significant drug interaction"
    assert data["P3489"]["label"] == "pregnancy category"
    assert data["P10946"]["label"] == "contraindicated in case of"
