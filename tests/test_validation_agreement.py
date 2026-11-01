"""Tests for scripts/validation_agreement.py on synthetic annotations (#77)."""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import validation_agreement as va  # noqa: E402

# 12 reference cells: 3 per language, mixed domains
REF = []
for lang in ["en", "he", "ar", "ru"]:
    for i, (dom, code) in enumerate([("g6pd", 2), ("teratogenicity", 1), ("qt", 0)]):
        REF.append({"drug": f"drug{i}", "lang": lang, "topic": dom, "final_code": code, "human_final_code": None})

# drug i is a positive for its own domain only (the risk column load_coded / filter_main_fit read)
DRUGS = pd.DataFrame([
    {"drug": "drug0", "category": "g6pd_only", "cpic_risk": "Contraindicated", "pregnancy_risk": "", "qt_risk": ""},
    {"drug": "drug1", "category": "teratogen_only", "cpic_risk": "", "pregnancy_risk": "Contraindicated", "qt_risk": ""},
    {"drug": "drug2", "category": "qt", "cpic_risk": "", "pregnancy_risk": "", "qt_risk": "Known risk"},
])


def _cell(r):
    return f"{r['drug']}__{r['lang']}__{r['topic']}"


def _write_annotator(path, name, overrides):
    """Annotator who copies the reference except for the overrides {cell_id: code}."""
    rows = [{"cell_id": _cell(r), "annotator": name, "code": str(overrides.get(_cell(r), r["final_code"])),
             "note": "", "saved_at": "2026-10-03T10:00:00+00:00"} for r in REF]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _setup(tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    pd.DataFrame(REF).to_csv(tmp_path / "ref.csv", index=False)
    DRUGS.to_csv(tmp_path / "drugs.csv", index=False)
    # Anna: higher than reference on he/g6pd (2 -> ... n/a), so use: he teratogenicity 1 -> 2 (LLM under-detected),
    # ar g6pd 2 -> 0 (LLM over-detected), ru qt marked "can't tell"
    _write_annotator(ann / "annotations_Anna.csv", "Anna", {
        "drug1__he__teratogenicity": 2, "drug0__ar__g6pd": 0, "drug2__ru__qt": "cant_tell"})
    _write_annotator(ann / "annotations_Ben.csv", "Ben", {"drug1__he__teratogenicity": 2})
    return ann, tmp_path / "ref.csv"


def test_vs_consensus_per_language_and_direction(tmp_path):
    ann, ref = _setup(tmp_path)
    df = va.attach_reference(va.load_marks(ann), ref)
    t = va.vs_consensus(df)
    anna = t[t["annotator"] == "Anna"].set_index(["language", "domain"])
    assert anna.loc[("en", "all"), "pct_exact_agree"] == 100.0 and anna.loc[("en", "all"), "weighted_kappa"] == 1.0
    assert anna.loc[("he", "all"), "n_exact_agree"] == 2 and anna.loc[("he", "all"), "n_marked"] == 3
    assert anna.loc[("ru", "all"), "n_cant_tell"] == 1 and anna.loc[("ru", "all"), "n_marked"] == 2
    assert anna.loc[("all", "all"), "n_marked"] == 11
    assert {"en", "he", "ar", "ru", "all"} == set(t["language"])

    d = va.direction(df)
    a = d[(d["annotator"] == "Anna") & (d["language"] == "all")].iloc[0]
    assert a["n_disagree"] == 2 and a["n_llm_under_detect"] == 1 and a["n_llm_over_detect"] == 1
    assert a["final1_annotator2"] == 1 and a["final2_annotator0"] == 1
    pooled = d[(d["annotator"] == "all_pooled") & (d["language"] == "he")].iloc[0]
    assert pooled["n_marked"] == 6 and pooled["n_llm_under_detect"] == 2  # Anna and Ben both marked he 1 -> 2


def test_between_annotators(tmp_path):
    ann, ref = _setup(tmp_path)
    t = va.between_annotators(va.attach_reference(va.load_marks(ann), ref))
    row = t[(t["annotator_a"] == "Anna") & (t["annotator_b"] == "Ben")].set_index("language")
    assert row.loc["he", "n_shared"] == 3 and row.loc["he", "pct_exact_agree"] == 100.0
    assert row.loc["ar", "n_exact_agree"] == 2 and row.loc["ar", "n_shared"] == 3
    assert row.loc["ru", "n_shared"] == 2  # Anna's "can't tell" is excluded
    assert not math.isnan(row.loc["all", "weighted_kappa"])


def test_cli_writes_all_tables_and_prints_the_outcome(tmp_path):
    ann, ref = _setup(tmp_path)
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, str(REPO / "scripts" / "validation_agreement.py"),
                          "--annotations-dir", str(ann), "--reference", str(ref),
                          "--drugs", str(tmp_path / "drugs.csv"), "--out-dir", str(out)],
                         check=True, capture_output=True, text=True)
    for name in ("agreement_vs_consensus.csv", "agreement_between_annotators.csv", "disagreement_direction.csv",
                 "agreement_by_language.csv", "sens_spec_by_language.csv", "impact_substitution.csv"):
        assert (out / name).stat().st_size > 0
    assert "Pre-specified outcome:" in run.stdout
    assert not (out / "impact_refit_hierarchical").exists()  # --refit is opt-in
    t = pd.read_csv(out / "agreement_vs_consensus.csv")
    assert {"annotator", "language", "domain", "n_marked", "weighted_kappa", "interpretable"} <= set(t.columns)
    assert (t["interpretable"] == (t["n_marked"] >= 8)).all() and t["interpretable"].any() and not t["interpretable"].all()
    assert (ann / "merged_item_level.csv").exists()  # item level stays in the ignored annotations dir


def test_progress_counts_marks_per_language(tmp_path):
    ann, _ = _setup(tmp_path)
    t = va.progress(va.load_marks(ann))
    assert t.loc["Anna", "total"] == 12 and t.loc["Ben", "he"] == 3


def test_notes_only_file_and_single_annotator_do_not_crash(tmp_path):
    import pytest
    ann = tmp_path / "ann"
    ann.mkdir()
    pd.DataFrame(REF).to_csv(tmp_path / "ref.csv", index=False)
    pd.DataFrame([{"cell_id": "drug0__en__g6pd", "annotator": "Zed", "code": "", "note": "x", "saved_at": ""}]
                 ).to_csv(ann / "annotations_Zed.csv", index=False)
    with pytest.raises(SystemExit):  # notes only: no marks to analyse yet
        va.load_marks(ann)
    _write_annotator(ann / "annotations_Anna.csv", "Anna", {})
    df = va.attach_reference(va.load_marks(ann), tmp_path / "ref.csv")
    between = va.between_annotators(df)
    assert between.empty and list(between.columns) == va.BETWEEN_COLUMNS  # one annotator: header only
    assert list(va.direction(df).columns) == va.DIRECTION_COLUMNS
