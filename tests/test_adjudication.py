"""Regression tests locking adjudication provenance integrity (audit issue #41).

No network. Works from the frozen coded artifacts only. Locks the invariants
verified in audit/rederive/rederive_adjudication_41.py:

  * every per-topic warning cell carries a KNOWN classification_method;
  * every escalated (genuinely disagreeing) cell is escalated, and every
    pharmacist-adjudicated cell was a genuine three-model disagreement;
  * the pharmacist's adjudicated codes are ingested verbatim into the per-topic
    files, and such cells are tagged human_review (not silently relabelled
    llm_consensus when their model codes were later collapsed);
  * the documented M3 (QT/PGx) provisional state: the adjudication worksheet is
    still empty and all 32 M3 disagreement cells resolve to majority_vote.

Once green, future edits cannot silently regress these.
"""

import csv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CODED = REPO / "data" / "coded"

CLASSIFICATIONS = CODED / "llm_classifications.csv"
DISAGREEMENTS = CODED / "llm_disagreements.csv"
WORKSHEET_M3 = CODED / "adjudication_worksheet_m3.csv"

WARNING_FILES = {
    "g6pd": CODED / "g6pd_warnings.csv",
    "teratogenicity": CODED / "teratogenicity_warnings.csv",
    "qt": CODED / "qt_warnings.csv",
    "pgx_scar": CODED / "pgx_scar_warnings.csv",
}
KNOWN_METHODS = {"llm_consensus", "human_review", "majority_vote"}
M3_TOPICS = {"qt", "pgx_scar"}


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _codes3(row, prefix="code_"):
    return [int(row[f"{prefix}{i}"]) for i in (1, 2, 3)]


def _genuine_disagreement(codes):
    valid = [c for c in codes if c >= 0]
    return len(set(valid)) > 1


@pytest.fixture(scope="module")
def warn():
    """(drug, lang, topic) -> (warning_code:int, classification_method:str)."""
    out = {}
    for topic, path in WARNING_FILES.items():
        for r in _load(path):
            out[(r["drug"], r["lang"], topic)] = (
                int(r["warning_code"]), r["classification_method"]
            )
    return out


@pytest.fixture(scope="module")
def adjudicated_keys():
    """Cells the pharmacist adjudicated, per the disagreement log."""
    return {
        (r["drug"], r["lang"], r["topic"])
        for r in _load(DISAGREEMENTS)
        if (r.get("human_code") or "").strip()
    }


def test_all_methods_known(warn):
    bad = {k: m for k, (_, m) in warn.items() if m not in KNOWN_METHODS}
    assert not bad, f"unknown classification_method: {bad}"


def test_every_nonconsensus_cell_is_escalated():
    dis_keys = {(r["drug"], r["lang"], r["topic"]) for r in _load(DISAGREEMENTS)}
    nonconsensus = {
        (r["drug"], r["lang"], r["topic"])
        for r in _load(CLASSIFICATIONS)
        if r["consensus"].lower() == "false"
    }
    assert nonconsensus - dis_keys == set(), (
        "genuine disagreement(s) missing from the escalation log"
    )


def test_only_genuine_disagreements_were_adjudicated():
    bad = [
        (r["drug"], r["lang"], r["topic"], _codes3(r))
        for r in _load(DISAGREEMENTS)
        if (r.get("human_code") or "").strip() and not _genuine_disagreement(_codes3(r))
    ]
    assert not bad, f"unanimous cells carry a human adjudication code: {bad}"


def test_adjudicated_codes_ingested_verbatim(warn, adjudicated_keys):
    """Each adjudicated code present in a per-topic file matches the human code."""
    human = {
        (r["drug"], r["lang"], r["topic"]): int(r["human_code"])
        for r in _load(DISAGREEMENTS)
        if (r.get("human_code") or "").strip()
    }
    checked = 0
    for k, hc in human.items():
        if k in warn:
            assert warn[k][0] == hc, f"adjudicated code not ingested for {k}"
            checked += 1
    assert checked == len(adjudicated_keys), "expected every adjudicated cell present"


def test_adjudicated_cells_tagged_human_review(warn, adjudicated_keys):
    """An adjudicated cell must be tagged human_review, never llm_consensus.

    Guards the merge_classifications precedence fix: a cell whose per-model codes
    were collapsed to a unanimous triple (consensus=true) but which was in fact
    pharmacist-adjudicated must still read as human_review.
    """
    mislabelled = [
        k for k in adjudicated_keys
        if k in warn and warn[k][1] != "human_review"
    ]
    assert not mislabelled, f"adjudicated cells not tagged human_review: {mislabelled}"


def test_human_review_count_matches_adjudications(warn, adjudicated_keys):
    n_human = sum(1 for _, m in warn.values() if m == "human_review")
    # Every per-topic-present adjudicated cell should be tagged human_review,
    # and nothing else should be.
    present = sum(1 for k in adjudicated_keys if k in warn)
    assert n_human == present == 8


def test_m3_adjudication_worksheet_still_empty():
    """Documented provisional state (issue #29): no QT/PGx adjudication ingested."""
    ws = _load(WORKSHEET_M3)
    assert len(ws) == 32
    assert all(not (r.get("human_code") or "").strip() for r in ws)
    assert {r["topic"] for r in ws} == M3_TOPICS


def test_m3_cells_resolve_to_majority_vote(warn):
    """All 32 QT/PGx disagreement cells await adjudication -> majority_vote."""
    n_major = sum(
        1 for (_, _, topic), (_, m) in warn.items()
        if topic in M3_TOPICS and m == "majority_vote"
    )
    n_human_m3 = sum(
        1 for (_, _, topic), (_, m) in warn.items()
        if topic in M3_TOPICS and m == "human_review"
    )
    assert n_major == 32
    assert n_human_m3 == 0
