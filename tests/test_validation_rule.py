"""Tests for the pre-specified validation rule (#77): src/g6pd/agreement.py and the new tables in
scripts/validation_agreement.py. Synthetic data only."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import validation_agreement as va  # noqa: E402
from g6pd import agreement as ag  # noqa: E402


def _cells(rows):
    """rows: (lang, domain, llm_code, human_code or None, disputed=False, n_cant_tell=0) -> cell_level frame."""
    recs = []
    for i, r in enumerate(rows):
        lang, dom, llm, human = r[:4]
        recs.append({"cell_id": f"d{i}__{lang}__{dom}", "drug": f"d{i}", "lang": lang, "domain": dom,
                     "ref_code": llm, "human_code": np.nan if human is None else float(human),
                     "disputed": r[4] if len(r) > 4 else False, "n_cant_tell": r[5] if len(r) > 5 else 0})
    return pd.DataFrame(recs).set_index("cell_id")


def test_thresholds_are_the_prespecified_ones():
    assert (ag.KAPPA_ADEQUATE, ag.KAPPA_STRONG, ag.ASYMMETRY_ALPHA) == (0.70, 0.80, 0.10)
    assert (ag.BOOTSTRAP_DRAWS, ag.MIN_STRATUM_N) == (2000, 8)
    assert ag.kappa_band(0.80, 5) == "strong" and ag.kappa_band(0.70, 5) == "adequate"
    assert ag.kappa_band(0.69, 5) == "below" and ag.kappa_band(float("nan"), 5) == "undefined"
    assert ag.kappa_band(float("nan"), 0) == "no data"


def test_kappa_is_pooled_across_domains_not_averaged():
    rows = ([("he", "g6pd", c, c) for c in (0, 1, 2, 2)]              # perfect in this domain
            + [("he", "qt", 0, 2), ("he", "qt", 2, 0), ("he", "qt", 1, 1), ("he", "qt", 0, 0)]
            + [("en", "qt", 1, 1)])
    t = va.by_language(_cells(rows)).set_index("language")
    he = [r for r in rows if r[0] == "he"]
    assert t.loc["he", "weighted_kappa"] == ag.weighted_kappa([r[2] for r in he], [r[3] for r in he])  # full precision
    assert t.loc["he", "n_cells"] == 8 and t.loc["he", "n_exact_agree"] == 6
    assert t.loc["he", "weighted_kappa"] < 1.0  # the qt disagreements count in the pooled figure
    assert set(t.index) == {"en", "he", "ar", "ru", "all"}  # languages without marks still get a row


def test_benchmark_is_applied_to_the_unrounded_kappa():
    assert ag.kappa_band(0.6996, 20) == "below"  # would print as 0.700
    assert ag.classify_outcome(_table(en=(20, 0.6996, "none"), he=(20, 0.9, "none")))[0] == "B"
    rng = np.random.default_rng(5)
    a, b = rng.integers(0, 3, 40), rng.integers(0, 3, 40)
    t = va.by_language(_cells([("he", "qt", int(x), int(y)) for x, y in zip(a, b)])).set_index("language")
    assert t.loc["he", "weighted_kappa"] == ag.weighted_kappa(a, b)  # the table is not rounded


def test_band_and_outcome_read_the_same_published_kappa():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a, b = rng.integers(0, 3, 20), rng.integers(0, 3, 20)
        t = va.by_language(_cells([("he", "qt", int(x), int(y)) for x, y in zip(a, b)]))
        r = t[t["language"] == "he"].iloc[0]
        assert r["kappa_band"] == ag.kappa_band(r["weighted_kappa"], r["n_cells"])  # band and CSV value agree


def test_languages_without_marks_are_no_data_not_passing():
    t = va.by_language(_cells([("en", "g6pd", 2, 2), ("en", "qt", 0, 0), ("en", "qt", 1, 1)])).set_index("language")
    for lang in ("he", "ar", "ru"):
        assert t.loc[lang, "n_cells"] == 0 and t.loc[lang, "kappa_band"] == "no data"
        assert t.loc[lang, "asymmetry"] == "no data" and math.isnan(t.loc[lang, "weighted_kappa"])


def test_bootstrap_is_deterministic_and_seed_dependent():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 30)
    b = np.where(rng.random(30) < 0.8, a, rng.integers(0, 3, 30))
    ci1, ci2 = ag.bootstrap_kappa_ci(a, b), ag.bootstrap_kappa_ci(a, b)
    assert ci1 == ci2
    assert ag.bootstrap_kappa_ci(a, b, seed=ag.BOOTSTRAP_SEED + 1) != ci1
    assert ci1[0] < ag.weighted_kappa(a, b) < ci1[1]
    t1 = va.by_language(_cells([("ar", "qt", int(x), int(y)) for x, y in zip(a, b)]))
    t2 = va.by_language(_cells([("ar", "qt", int(x), int(y)) for x, y in zip(a, b)]))
    pd.testing.assert_frame_equal(t1, t2)


def test_bootstrap_degenerate_inputs_give_nan_not_errors():
    assert all(math.isnan(x) for x in ag.bootstrap_kappa_ci([], []))
    assert all(math.isnan(x) for x in ag.bootstrap_kappa_ci([2, 2, 2], [2, 2, 2]))  # kappa undefined everywhere


def test_sens_spec_edge_cases_give_nan_not_errors():
    empty = ag.sens_spec([], [])
    assert empty["n_pos"] == 0 and empty["n_neg"] == 0
    assert all(math.isnan(empty[k]) for k in ("sensitivity", "specificity", "sensitivity_ci_low", "specificity_ci_high"))
    only_pos = ag.sens_spec([True, True, True], [True, True, False])
    assert only_pos["n_neg"] == 0 and math.isnan(only_pos["specificity"]) and math.isnan(only_pos["specificity_ci_low"])
    assert only_pos["sensitivity"] == pytest.approx(2 / 3)
    perfect = ag.sens_spec([True, False], [True, False])  # exact CI is wide at n = 1, upper bound 1
    assert perfect["sensitivity"] == 1.0 and perfect["sensitivity_ci_high"] == 1.0 and perfect["sensitivity_ci_low"] < 0.1
    lo, hi = ag.clopper_pearson(5, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.187, 0.813)  # textbook exact interval for 5/10


def test_sens_spec_table_definitions():
    # human vs LLM: mention = 0 vs >=1; directive = 2 vs 1, only among cells where both codes are >=1
    rows = [("he", "g6pd", 2, 2), ("he", "g6pd", 1, 2),   # directive: one hit, one miss (LLM said 1)
            ("he", "qt", 1, 1), ("he", "qt", 2, 1),       # directive negatives: one correct, one false alarm
            ("he", "qt", 0, 1),                           # LLM misses a mention (not a directive cell)
            ("he", "qt", 1, 0)]                           # LLM false mention
    t = va.sens_spec_table(_cells(rows)).set_index(["language", "decision"])
    m, d = t.loc[("he", "mention")], t.loc[("he", "directive")]
    assert (m["n_pos"], m["n_neg"], m["sensitivity"], m["specificity"]) == (5, 1, 0.8, 0.0)
    assert (d["n"], d["n_pos"], d["n_neg"], d["sensitivity"], d["specificity"]) == (4, 2, 2, 0.5, 0.5)
    assert d["sensitivity_ci_low"] < 0.5 < d["sensitivity_ci_high"]
    assert math.isnan(t.loc[("ru", "directive"), "sensitivity"]) and t.loc[("ru", "mention"), "n"] == 0


def test_sign_test_and_asymmetry_criterion():
    assert math.isnan(ag.sign_test_p(0, 0))
    assert ag.sign_test_p(8, 0) == pytest.approx(2 * 0.5 ** 8)
    assert ag.sign_test_p(3, 3) == 1.0
    assert ag.sign_test_p(1, 5) == ag.sign_test_p(5, 1)  # two-sided
    assert ag.asymmetry(5, 0, 12) == "under"      # p = 0.0625 < 0.10, LLM below human
    assert ag.asymmetry(4, 0, 12) == "none"       # p = 0.125, not enough
    assert ag.asymmetry(0, 5, 12) == "over"       # mirror image: informational only
    assert ag.asymmetry(0, 0, 12) == "none" and ag.asymmetry(0, 0, 0) == "no data"


def test_by_language_bias_and_direction():
    rows = [("ar", "g6pd", 0, 2), ("ar", "g6pd", 1, 2), ("ar", "qt", 0, 1), ("ar", "qt", 1, 2),
            ("ar", "qt", 0, 1), ("ar", "qt", 2, 2), ("ar", "qt", 1, 0)]
    ar = va.by_language(_cells(rows)).set_index("language").loc["ar"]
    assert ar["n_llm_under"] == 5 and ar["n_llm_over"] == 1 and ar["n_cells"] == 7
    assert ar["mean_bias_human_minus_llm"] == pytest.approx((2 + 1 + 1 + 1 + 1 + 0 - 1) / 7, abs=1e-3)
    assert ar["share_llm_under"] == pytest.approx(5 / 7, abs=1e-3)
    assert ar["sign_test_p"] == pytest.approx(ag.sign_test_p(5, 1), abs=1e-4)
    assert ar["asymmetry"] == "none"  # 5 vs 1: p = 0.219


def _table(**langs):
    """language -> (n_cells, kappa, asymmetry)"""
    return pd.DataFrame([{"language": lang, "n_cells": v[0], "weighted_kappa": v[1], "asymmetry": v[2]}
                         for lang, v in langs.items()])


def test_outcome_a_b_c_and_no_data():
    ok = (20, 0.85, "none")
    assert ag.classify_outcome(_table(en=ok, he=ok, ar=(18, 0.72, "none"), ru=ok))[0] == "A"
    # A requires every language WITH data to pass; a language without marks is reported, not counted as passing
    outcome, note = ag.classify_outcome(_table(en=ok, he=(0, float("nan"), "no data"), ar=ok, ru=(0, float("nan"), "no data")))
    assert outcome == "A" and "no data: he, ru" in note
    assert ag.classify_outcome(_table(en=ok, he=ok, ar=(18, 0.55, "none"), ru=ok))[0] == "B"
    assert ag.classify_outcome(_table(en=ok, he=ok, ar=(18, 0.55, "under"), ru=ok))[0] == "C"
    assert ag.classify_outcome(_table(en=ok, he=(18, 0.9, "under"), ar=ok, ru=ok))[0] == "C"  # even with high kappa
    none = (0, float("nan"), "no data")
    assert ag.classify_outcome(_table(en=none, he=none, ar=none, ru=none))[0] == "no data"
    assert ag.classify_outcome(_table(en=none, he=none, ar=none, ru=none).iloc[:0])[0] == "no data"


def test_outcome_outside_the_prespecified_patterns_is_not_forced():
    ok = (20, 0.85, "none")
    # asymmetry in English as well: not C (the comparator shows it too); not A or B either
    assert ag.classify_outcome(_table(en=(20, 0.85, "under"), he=ok, ar=(18, 0.6, "under"), ru=ok))[0] == "none of A/B/C"
    assert ag.classify_outcome(_table(en=ok, he=ok, ar=ok, ru=(20, 0.85, "under")))[0] == "none of A/B/C"  # Russian only
    assert ag.classify_outcome(_table(en=ok, he=ok, ar=(18, float("nan"), "none"), ru=ok))[0] == "none of A/B/C"
    # the over-detecting mirror image is informational: it does not block A
    assert ag.classify_outcome(_table(en=ok, he=ok, ar=(18, 0.9, "over"), ru=ok))[0] == "A"
    # C needs English data: "not in English" cannot be established without it
    outcome, note = ag.classify_outcome(_table(en=(0, float("nan"), "no data"), ar=(18, 0.5, "under")))
    assert outcome == "none of A/B/C" and "en has no data, so C cannot be established" in note


def test_cell_level_collapses_raters():
    base = {"drug": "d", "lang": "ar", "domain": "qt", "ref_code": 1}
    def cell(cell_id, *marks):
        return [{**base, "cell_id": cell_id, "annotator": a, "cant_tell": isinstance(c, str),
                 "ann_code": np.nan if isinstance(c, str) else float(c)} for a, c in marks]
    df = pd.DataFrame(cell("agree", ("A", 2), ("B", 2)) + cell("dispute", ("A", 0), ("B", 2))
                      + cell("one_cant", ("A", "x"), ("B", 1)) + cell("only_cant", ("A", "x")) + cell("single", ("A", 0)))
    c = va.cell_level(df)
    assert c.loc["agree", "human_code"] == 2 and not c.loc["agree", "disputed"]
    assert math.isnan(c.loc["dispute", "human_code"]) and c.loc["dispute", "disputed"]
    assert c.loc["one_cant", "human_code"] == 1 and c.loc["one_cant", "n_cant_tell"] == 1
    assert math.isnan(c.loc["only_cant", "human_code"]) and not c.loc["only_cant", "disputed"]
    assert c.loc["single", "human_code"] == 0
    t = va.by_language(c).set_index("language").loc["ar"]
    assert (t["n_cells"], t["n_disputed"], t["n_cant_tell"]) == (3, 1, 2)  # disputed cell excluded but counted


def _coded():
    rows = []
    for lang, codes in {"en": [2, 2, 2, 1], "he": [2, 1, 0, 0], "ar": [1, 1, 0, 0], "ru": [2, 2, 1, 0]}.items():
        for i, c in enumerate(codes):
            rows.append({"drug": f"d{i}", "lang": lang, "topic": "g6pd", "code": c})
    return pd.DataFrame(rows)


def test_substitution_and_table4_shares():
    # Arabic: human says d0 is a directive (LLM 1), d2 is mentioned (LLM 0). d1 is disputed, d3 'cant_tell'.
    cells = pd.DataFrame([
        {"cell_id": "d0__ar__g6pd", "drug": "d0", "lang": "ar", "domain": "g6pd", "ref_code": 1, "human_code": 2.0, "disputed": False, "n_cant_tell": 0},
        {"cell_id": "d1__ar__g6pd", "drug": "d1", "lang": "ar", "domain": "g6pd", "ref_code": 1, "human_code": np.nan, "disputed": True, "n_cant_tell": 0},
        {"cell_id": "d2__ar__g6pd", "drug": "d2", "lang": "ar", "domain": "g6pd", "ref_code": 0, "human_code": 1.0, "disputed": False, "n_cant_tell": 0},
        {"cell_id": "d3__ar__g6pd", "drug": "d3", "lang": "ar", "domain": "g6pd", "ref_code": 0, "human_code": np.nan, "disputed": False, "n_cant_tell": 1},
        {"cell_id": "d0__en__g6pd", "drug": "d0", "lang": "en", "domain": "g6pd", "ref_code": 2, "human_code": 2.0, "disputed": False, "n_cant_tell": 0},
    ]).set_index("cell_id")
    sub = va.substitute(_coded(), cells)
    ar = sub[sub["lang"] == "ar"].set_index("drug")
    assert ar["code"].tolist() == [2, 1, 1, 0] and ar["llm_code"].tolist() == [1, 1, 0, 0]  # d1 (disputed), d3 untouched
    assert ar["substituted"].tolist() == [True, False, True, False]
    assert sub[sub["lang"] == "he"]["substituted"].sum() == 0  # unvalidated languages are never touched

    t = va.impact_table(_coded(), cells).set_index(["version", "language"])
    assert (t.loc[("llm_final", "ar"), "n_mentioned"], t.loc[("llm_final", "ar"), "n_directive"]) == (2, 0)
    assert (t.loc[("human_substituted", "ar"), "n_mentioned"], t.loc[("human_substituted", "ar"), "n_directive"]) == (3, 1)
    assert t.loc[("human_substituted", "ar"), "p_directive_given_mention"] == pytest.approx(1 / 3, abs=1e-3)
    r = t.loc[("human_substituted", "ar")]
    assert (r["n_substituted"], r["n_changed"], r["n_disputed_excluded"], r["n_cant_tell_only"]) == (2, 2, 1, 1)
    assert t.loc[("human_substituted", "en"), "n_changed"] == 0
    assert t.loc[("llm_final", "en"), "or_direction"] == "reference"
    assert pd.isna(t.loc[("llm_final", "ar"), "n_substituted"])
    # unchanged languages give identical shares in both versions
    for col in ("n_mentioned", "n_directive"):
        assert t.loc[("llm_final", "ru"), col] == t.loc[("human_substituted", "ru"), col]


def test_pooled_or_direction_and_degenerate_fit():
    big = pd.concat([_coded().assign(drug=lambda d: d["drug"] + str(k)) for k in range(5)], ignore_index=True)
    ors = va.pooled_ors_fast(big)
    assert set(ors) == {"he", "ar", "ru"} and all(o[0] < 1 for o in ors.values())  # English warns more
    assert all(o[1] < o[0] < o[2] for o in ors.values())
    no_en = va.pooled_ors_fast(big[big["lang"] != "en"])
    assert all(math.isnan(x) for o in no_en.values() for x in o)  # no English reference: undefined, not an error


def test_write_substituted_csv_feeds_the_hierarchical_script(tmp_path):
    ref = pd.DataFrame([{"drug": "d0", "lang": "ar", "topic": "g6pd", "final_code": 1, "human_final_code": np.nan},
                        {"drug": "d1", "lang": "ar", "topic": "g6pd", "final_code": 1, "human_final_code": 0.0},
                        {"drug": "d2", "lang": "ar", "topic": "g6pd", "final_code": 1, "human_final_code": np.nan}])
    ref.to_csv(tmp_path / "ref.csv", index=False)
    cells = pd.DataFrame([{"cell_id": "d0__ar__g6pd", "drug": "d0", "lang": "ar", "domain": "g6pd", "ref_code": 1,
                           "human_code": 2.0, "disputed": False, "n_cant_tell": 0},
                          {"cell_id": "d2__ar__g6pd", "drug": "d2", "lang": "ar", "domain": "g6pd", "ref_code": 1,
                           "human_code": np.nan, "disputed": True, "n_cant_tell": 0}]).set_index("cell_id")
    out = pd.read_csv(va.write_substituted_csv(tmp_path / "ref.csv", cells, tmp_path / "sub.csv")).set_index("drug")
    assert out.loc["d0", "human_final_code"] == 2 and out.loc["d1", "human_final_code"] == 0  # existing adjudication kept
    assert math.isnan(out.loc["d2", "human_final_code"])  # disputed cells are not substituted
    assert list(out.columns) == ["lang", "topic", "final_code", "human_final_code"]
