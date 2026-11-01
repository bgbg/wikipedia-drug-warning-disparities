"""Tests for the three-domain agreement table and the revision matcher (#79). No network, no MCMC."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import fetch_revision_ids as fri  # noqa: E402
import model_agreement_table as mat  # noqa: E402


def test_agreement_table_totals_and_committed_csv():
    table = mat.agreement_table(mat.classified_cells())
    row = table[(table["language"] == "all") & (table["domain"] == "all")].iloc[0]
    assert (row["n_cells"], row["n_unanimous"]) == (184, 160)
    # per-domain rows add up to the overall row
    by_domain = table[(table["language"] == "all") & (table["domain"] != "all")]
    assert by_domain["n_cells"].sum() == 184 and by_domain["n_unanimous"].sum() == 160
    committed = pd.read_csv(mat.OUT)
    pd.testing.assert_frame_equal(table.reset_index(drop=True), committed, check_dtype=False)


def test_classify_revisions():
    stored = fri.sha("text")
    one = {"revid": 1, "timestamp": "2026-01-01T00:00:00Z", "sha": stored}
    other = {"revid": 2, "timestamp": "2026-01-01T01:00:00Z", "sha": fri.sha("other")}
    assert fri.classify([other, one], stored)["revision_id"] == 1  # exact text match picks the right revision
    assert fri.classify([other], stored)["status"] == "date_only"  # one revision, text differs: no ID claimed
    assert fri.classify([other, other], stored)["status"] == "ambiguous"
    assert fri.classify([], stored)["status"] == "not_found"
    assert fri.sha(" text\n") == stored  # surrounding whitespace is ignored


def test_committed_snapshot_has_an_id_for_every_panel_article():
    snap = pd.read_csv(REPO / "data" / "snapshot_revisions.csv")
    assert len(snap) == 118 and (snap["status"] == "exact_match").all()
    assert snap["revision_id"].notna().all() and not snap.duplicated(["lang", "drug"]).any()
