"""Offline tests for the Wikidata audit helpers (issue #76).

No network: SPARQL builders, topic-to-property mapping, denominator rule and
claim parsing are pure functions.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from g6pd import wikidata  # noqa: E402
from g6pd.wikidata import (  # noqa: E402
    CONTEXT_PROPS, build_atc_count_query, build_statement_count_query, claim_values,
    in_denominator, matching_values, topic_checks,
)


def test_atc_universe_query():
    q = build_atc_count_query()
    assert "COUNT(DISTINCT ?s)" in q and "wdt:P267" in q
    assert "VALUES" not in q


def test_atc_query_with_property_and_targets():
    q = build_atc_count_query("P10946", {"Q653924", "Q1625433"})
    assert "wdt:P267" in q and "wdt:P10946" in q
    assert "VALUES ?o { wd:Q1625433 wd:Q653924 }" in q  # sorted, deterministic


def test_atc_query_property_without_targets_has_no_values_clause():
    q = build_atc_count_query("P3489")
    assert "wdt:P3489" in q and "VALUES" not in q


@pytest.mark.parametrize("prop,targets", [("P10946; DROP", None), ("10946", None), ("P3489", {"Q1 }"})])
def test_atc_query_rejects_malformed_ids(prop, targets):
    with pytest.raises(ValueError):
        build_atc_count_query(prop, targets)


def test_primary_properties_per_topic():
    primary = {t: {p for p, r, _ in topic_checks(t) if r == "primary"}
               for t in ("g6pd", "teratogenicity", "qt")}
    assert primary == {"g6pd": {"P10946"}, "qt": {"P10946"}, "teratogenicity": {"P10946", "P3489"}}


def test_topic_targets():
    by = {t: {(p, frozenset(tg) if tg else None) for p, r, tg in topic_checks(t) if r == "primary"}
          for t in ("g6pd", "teratogenicity", "qt")}
    assert by["g6pd"] == {("P10946", frozenset({"Q848343"}))}
    assert by["qt"] == {("P10946", frozenset({"Q653924", "Q1625433"}))}
    assert by["teratogenicity"] == {("P10946", frozenset({"Q11995"})), ("P3489", None)}


def test_p1909_and_p769_are_context_only():
    assert CONTEXT_PROPS == ("P1909", "P769")
    for t in ("g6pd", "teratogenicity", "qt"):
        roles = {p: r for p, r, _ in topic_checks(t)}
        assert roles["P1909"] == roles["P769"] == "context"


@pytest.mark.parametrize("topic,basis,expected", [
    ("g6pd", "Contraindicated", True),
    ("g6pd", "Avoid in deficient patients", False),
    ("g6pd", "Use with caution", False),
    ("teratogenicity", "Contraindicated (iPLEDGE/REMS)", True),
    ("qt", "Known risk of TdP (CredibleMeds)", False),
    ("qt", "Contraindicated", False),  # QT is always reported separately
])
def test_denominator_rule(topic, basis, expected):
    assert in_denominator(topic, basis) is expected


def _stmt(value, rank="normal"):
    return {"rank": rank, "mainsnak": {"datavalue": {"value": value}}}


def test_claim_values_skips_deprecated_and_handles_strings():
    claims = {
        "P3489": [_stmt({"id": "Q1"}), _stmt({"id": "Q2"}, "deprecated"),
                  {"rank": "normal", "mainsnak": {"snaktype": "novalue"}}],
        "P267": [_stmt("B03AA")],
    }
    assert claim_values(claims, ("P3489", "P267", "P10946")) == {"P3489": ["Q1"], "P267": ["B03AA"]}


def test_claim_values_prefers_preferred_rank_like_wdt():
    """wdt: returns only preferred statements when any exist, else the normal ones."""
    claims = {
        "P3489": [_stmt({"id": "Q1"}), _stmt({"id": "Q2"}, "preferred"), _stmt({"id": "Q3"}, "deprecated")],
        "P10946": [_stmt({"id": "Q4"}), _stmt({"id": "Q5"})],
    }
    assert claim_values(claims, ("P3489", "P10946")) == {"P3489": ["Q2"], "P10946": ["Q4", "Q5"]}


def test_matching_values():
    assert matching_values(["Q1", "Q2"], None) == ["Q1", "Q2"]
    assert matching_values(["Q1", "Q2"], {"Q2", "Q9"}) == ["Q2"]
    assert matching_values([], {"Q2"}) == []


def test_statement_count_query():
    q = build_statement_count_query("P10946")
    assert "p:P10946" in q and "COUNT(*)" in q
    with pytest.raises(ValueError):
        build_statement_count_query("10946")


class _FakeResp:
    def __init__(self, body, status=200):
        self.status_code, self._body = status, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


def test_api_error_body_with_http_200_fails_loudly(monkeypatch):
    monkeypatch.setattr(wikidata.requests, "get",
                        lambda *a, **k: _FakeResp({"error": {"code": "internal_api_error"}}))
    with pytest.raises(RuntimeError, match="Wikidata API error"):
        wikidata.fetch_drug_claims(["Q1"], ("P10946",))


def test_missing_requested_entity_fails_loudly(monkeypatch):
    monkeypatch.setattr(wikidata.requests, "get", lambda *a, **k: _FakeResp(
        {"entities": {"Q1": {"claims": {}}, "Q2": {"missing": ""}}}))
    with pytest.raises(RuntimeError, match="no entity"):
        wikidata.fetch_drug_claims(["Q1", "Q2"], ("P10946",))
    with pytest.raises(RuntimeError, match="no entity"):
        wikidata.fetch_drug_claims(["Q1", "Q3"], ("P10946",))
