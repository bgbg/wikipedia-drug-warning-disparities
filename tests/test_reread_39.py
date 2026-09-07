"""Structural regression test for the #39 warning-code re-read audit.

This is a *judgement* audit (Claude re-read the codes from frozen article text),
so the test only locks the structure and internal consistency of the result file
``audit/rederive/reread_39.csv`` and its alignment with the recorded codes in
``data/coded/coded_four_topic.csv``. It does NOT assert any specific re-read code
(those are flagged for clinical re-judgement, never overridden).
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REREAD = ROOT / "audit" / "rederive" / "reread_39.csv"
CODED = ROOT / "data" / "coded" / "coded_four_topic.csv"

VALID_CODES = {0, 1, 2}
VALID_LOCATIONS = {
    "lead", "contraindications_section", "adverse_effects_section",
    "side_effects", "black_box_warning", "warnings", "body",
    "footnote-only", "absent",
}
EXPECTED_COLS = [
    "drug", "lang", "topic", "recorded_code", "reread_code",
    "recorded_location", "reread_location", "agree", "note",
]


@pytest.fixture(scope="module")
def reread():
    # PGx-SCAR dropped as an analysed topic 2026-07-01; compare only the three
    # analysed topics (the frozen reread CSV keeps its original four-topic rows).
    df = pd.read_csv(REREAD)
    return df[df["topic"] != "pgx_scar"].reset_index(drop=True)


def test_columns(reread):
    assert list(reread.columns) == EXPECTED_COLS


def test_one_row_per_recorded_cell(reread):
    coded = pd.read_csv(CODED)
    assert len(reread) == len(coded)
    rkeys = set(map(tuple, reread[["drug", "lang", "topic"]].values))
    ckeys = set(map(tuple, coded[["drug", "lang", "topic"]].values))
    assert rkeys == ckeys
    # no duplicate cells
    assert not reread.duplicated(subset=["drug", "lang", "topic"]).any()


def test_codes_in_range(reread):
    assert set(reread["recorded_code"].unique()) <= VALID_CODES
    assert set(reread["reread_code"].unique()) <= VALID_CODES


def test_locations_in_vocab(reread):
    rr = set(reread["reread_location"].dropna().unique())
    assert rr <= VALID_LOCATIONS, rr - VALID_LOCATIONS


def test_recorded_code_matches_source(reread):
    coded = pd.read_csv(CODED)[["drug", "lang", "topic", "final_code"]]
    m = reread.merge(coded, on=["drug", "lang", "topic"], how="left")
    assert m["final_code"].notna().all()
    assert (m["recorded_code"] == m["final_code"]).all()


def test_agree_flag_consistent(reread):
    expected = reread["recorded_code"] == reread["reread_code"]
    assert (reread["agree"].astype(bool) == expected).all()


def test_absent_iff_code_zero_for_reread(reread):
    # reread location "absent" must coincide with reread code 0
    absent = reread["reread_location"] == "absent"
    assert (reread.loc[absent, "reread_code"] == 0).all()


def test_every_cell_has_note(reread):
    assert reread["note"].notna().all()
    assert (reread["note"].astype(str).str.len() > 0).all()
