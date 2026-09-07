"""Regression tests locking the deterministic statistical quantities (audit #42).

No network, no MCMC. These re-derive — with an implementation independent of
src/g6pd/ordinal_analysis.py where practical — the model-free numbers reported
in Results §5 and Appendix B from the frozen data/coded/coded_four_topic.csv:

  * {0,1,2} code composition per (language, topic) — Results §5 table.
  * Cliff's delta point estimate sign + magnitude (EN vs each non-EN edition,
    per topic) — Results §5 table.
  * Paired Wilcoxon non-zero-pair counts and p-values — Appendix B table.
  * Coverage / cell counts (8 drugs per topic, 4 langs).

Once green, future edits cannot silently regress these verified results.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from g6pd import ordinal_analysis as oa  # noqa: E402

CODED = REPO / "data" / "coded" / "coded_four_topic.csv"
DRUGS = REPO / "data" / "drugs.csv"


@pytest.fixture(scope="module")
def main_fit():
    df = oa.load_coded(CODED, DRUGS)
    return oa.filter_main_fit(df)


# --- Results §5 / composition: every topic has 8 drugs x 4 langs ----------------
def test_main_fit_shape(main_fit):
    # PGx-SCAR dropped 2026-07-01: three analysed topics, 8 positives x 4 langs each.
    assert len(main_fit) == 96  # 8 drugs x 3 topics x 4 langs
    by_topic = main_fit.groupby("topic")["drug"].nunique().to_dict()
    assert by_topic == {"g6pd": 8, "teratogenicity": 8, "qt": 8}


# Expected composition (% code 0/1/2) — Results §5 table (g6pd + teratogenicity)
# and the figures for the two new domains. Percentages rounded to nearest int.
EXPECTED_COMPOSITION = {
    ("en", "g6pd"): (12, 12, 75),
    ("ru", "g6pd"): (75, 0, 25),
    ("ar", "g6pd"): (50, 25, 25),
    ("he", "g6pd"): (25, 25, 50),
    ("en", "teratogenicity"): (0, 25, 75),
    ("ru", "teratogenicity"): (25, 25, 50),
    ("ar", "teratogenicity"): (0, 50, 50),
    ("he", "teratogenicity"): (12, 38, 50),
}


@pytest.mark.parametrize("key,expected", list(EXPECTED_COMPOSITION.items()))
def test_composition(main_fit, key, expected):
    lg, tp = key
    cell = main_fit[(main_fit.lang == lg) & (main_fit.topic == tp)]
    assert len(cell) == 8
    pct = (
        round(100 * (cell.code == 0).mean()),
        round(100 * (cell.code == 1).mean()),
        round(100 * (cell.code == 2).mean()),
    )
    assert pct == expected


# --- Cliff's delta point estimate (Results §5: EN vs non-EN, positive=EN warns more)
def _cliffs_direct(x, y):
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    return (gt - lt) / (len(x) * len(y))


# Results §5 Cliff's delta point estimates (g6pd + teratogenicity)
EXPECTED_CLIFFS = {
    ("ru", "g6pd"): 0.59,
    ("ar", "g6pd"): 0.53,
    ("he", "g6pd"): 0.25,
    ("ru", "teratogenicity"): 0.31,
    ("ar", "teratogenicity"): 0.25,
    ("he", "teratogenicity"): 0.28,
}


@pytest.mark.parametrize("key,expected", list(EXPECTED_CLIFFS.items()))
def test_cliffs_delta_point(main_fit, key, expected):
    lg, tp = key
    sub = main_fit[(main_fit.topic == tp) & (main_fit.lang.isin(("en", lg)))]
    piv = sub.pivot_table(index="drug", columns="lang", values="code",
                          aggfunc="first").dropna(subset=["en", lg])
    en = piv["en"].values.astype(float)
    other = piv[lg].values.astype(float)
    d_direct = _cliffs_direct(en, other)
    d_prod = oa.cliffs_delta(en, other)
    # independent double-loop agrees with production routine
    assert abs(d_direct - d_prod) < 1e-9
    # positive => EN warns more, and matches the manuscript value
    assert d_direct > 0
    assert round(d_direct, 2) == expected


# --- Paired Wilcoxon (Appendix B: non-zero pairs + p) ---------------------------
EXPECTED_WILCOXON = {
    ("ru", "g6pd"): (5, 0.0625),
    ("ar", "g6pd"): (4, 0.125),
    ("he", "g6pd"): (4, 0.500),
    ("ru", "teratogenicity"): (3, 0.250),
    ("ar", "teratogenicity"): (2, 0.500),
    ("he", "teratogenicity"): (3, 0.250),
}


def test_wilcoxon(main_fit):
    w = oa.paired_wilcoxon_table(main_fit).set_index(["language", "topic"])
    for (lg, tp), (n_nz, p) in EXPECTED_WILCOXON.items():
        row = w.loc[(lg, tp)]
        assert int(row["n_nonzero_diffs"]) == n_nz
        assert row["pvalue"] == pytest.approx(p, abs=1e-6)
