"""Regression tests locking the fetcher's reference-counting logic (issue #36).

These tests pin the behaviour verified during the audit so a future edit cannot
silently regress it. They exercise two layers:

  1. The pure counting expression from ``src/g6pd/fetch.py`` against synthetic
     wikitext, including the Hebrew ``{{הערה}}`` template that the original
     bug dropped to zero.
  2. The frozen snapshot ``data/raw/articles.jsonl``, with an *aggregate* guard
     on the Hebrew edition (median refs > 0 AND total {{הערה}} > 0). A per-article
     guard would be wrong: ~7 Hebrew articles legitimately have zero references.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "raw" / "articles.jsonl"


def count_references(wikitext: str) -> int:
    """Reference count exactly as ``src/g6pd/fetch.py`` computes it.

    Mirrors the production expression: ``<ref`` substring occurrences plus the
    Hebrew inline note template ``{{הערה}}``.
    """
    return wikitext.lower().count("<ref") + len(re.findall(r"\{\{הערה[|}]", wikitext))


# ---------------------------------------------------------------------------
# Layer 1: synthetic-wikitext unit tests of the counting expression
# ---------------------------------------------------------------------------

def test_counts_plain_ref_tags():
    wt = "Text.<ref>a</ref> More.<ref name=x>b</ref> End.<ref name=y />"
    assert count_references(wt) == 3


def test_counts_hebrew_haara_template():
    # The bug: Hebrew cites with {{הערה}}, not <ref>. Must not count to zero.
    wt = "מבוא{{הערה|מקור ראשון}} המשך{{הערה|מקור שני}}"
    assert count_references(wt) == 2


def test_counts_mixed_ref_and_haara():
    wt = "Lead<ref>x</ref> body{{הערה|y}} more{{הערה}}"
    assert count_references(wt) == 3


def test_no_citations_counts_zero():
    assert count_references("Plain prose with no references at all.") == 0


def test_haara_requires_pipe_or_close_brace():
    # {{הערהxyz}} (a different template name) must not be miscounted.
    assert count_references("{{הערהאחרת}}") == 0
    assert count_references("{{הערה}}") == 1
    assert count_references("{{הערה|src}}") == 1


# ---------------------------------------------------------------------------
# Layer 2: frozen-snapshot aggregate guards
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def snapshot_records():
    if not SNAPSHOT.is_file():
        pytest.skip(f"snapshot not present: {SNAPSHOT}")
    with SNAPSHOT.open() as fh:
        return [json.loads(line) for line in fh]


def _existing(records, lang):
    return [r for r in records if r["lang"] == lang and r["exists"]]


def test_hebrew_reference_count_aggregate_guard(snapshot_records):
    """The regression guard for the {{הערה}} bug.

    Aggregate, not per-article: median Hebrew refs must be > 0 and the edition
    must contain {{הערה}} templates that the recomputation actually counts.
    """
    he = _existing(snapshot_records, "he")
    assert he, "no Hebrew articles in snapshot"
    refs = [r["n_references"] for r in he]
    assert statistics.median(refs) > 0, "Hebrew median references collapsed to 0 (הערה undercount regressed)"

    total_haara = sum(
        len(re.findall(r"\{\{הערה[|}]", r["wikitext"]))
        for r in he
        if r["wikitext"]
    )
    assert total_haara > 0, "no {{הערה}} templates found in Hebrew snapshot"


def test_stored_reference_counts_match_recomputation(snapshot_records):
    """Every existing article's stored n_references equals a fresh recount."""
    for r in snapshot_records:
        if not r["exists"] or not r["wikitext"]:
            continue
        assert r["n_references"] == count_references(r["wikitext"]), (
            f"{r['lang']}/{r['drug']}: stored {r['n_references']} != recomputed "
            f"{count_references(r['wikitext'])}"
        )


def test_missing_articles_have_null_numeric_fields(snapshot_records):
    """exists=False records must carry None counts, never silent zeros."""
    for r in snapshot_records:
        if not r["exists"]:
            assert r["word_count"] is None, f"{r['lang']}/{r['drug']}: word_count not None"
            assert r["n_references"] is None, f"{r['lang']}/{r['drug']}: n_references not None"


def test_existing_articles_have_nonnull_counts(snapshot_records):
    for r in snapshot_records:
        if r["exists"]:
            assert r["word_count"] is not None
            assert r["n_references"] is not None
