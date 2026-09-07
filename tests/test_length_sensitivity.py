"""Regression tests for the length-confound sensitivity analysis (#65).

Locks the *stored* outputs of `scripts/length_sensitivity.py` under
`data/coded/length_sensitivity/`. No refit and no network: the test reads the
frozen CSVs, so it pins the reported claims rather than re-running MCMC.

The claims being locked:
  * log(word count) is a strong positive predictor of warning strength — the
    reviewer's length objection is real and is not swept under the rug.
  * Adjusting for length attenuates the language effect (it is a *mediator*),
    so the length-adjusted model must NOT be read as the headline.
  * Conditional on the risk being mentioned, Arabic still withholds the
    directive (HDI excludes 1) while Russian does not differ from English —
    the length-independent core of the reframed thesis.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "coded" / "length_sensitivity"


@pytest.fixture(scope="module")
def pooled():
    return pd.read_csv(OUT / "pooled_ors_by_model.csv")


@pytest.fixture(scope="module")
def conditional():
    return pd.read_csv(OUT / "conditional_ors.csv").set_index("language")


@pytest.fixture(scope="module")
def counts():
    return pd.read_csv(OUT / "conditional_counts.csv").set_index("language")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((OUT / "manifest.json").read_text())


def test_outputs_exist():
    for fn in ("pooled_ors_by_model.csv", "log_wc_effect.csv",
               "conditional_ors.csv", "conditional_counts.csv", "manifest.json"):
        assert (OUT / fn).is_file(), f"missing {fn}"


def test_manifest_shape(manifest):
    # 96 main-fit rows (8 positives x 3 topics x 4 langs); 85 have an existing
    # article; 70 of the 96 mention the risk at all.
    assert manifest["n_main_fit_rows"] == 96
    assert manifest["n_existing_articles"] == 85
    assert manifest["n_mentioned_cells"] == 70
    assert manifest["seed"] == 23


def test_length_is_a_strong_predictor():
    """The reviewer's objection is real: longer articles warn more strongly."""
    r = pd.read_csv(OUT / "log_wc_effect.csv").iloc[0]
    assert r["or_median"] == pytest.approx(3.60, abs=0.15)
    assert r["hdi_low"] > 1.0, "log(word count) HDI must exclude 1 from below"
    assert r["prob_beta_gt_0"] > 0.99


def test_adjusting_for_length_attenuates_the_language_effect(pooled):
    """Length is a mediator: adjusting for it moves every edition toward 1."""
    base = pooled[pooled["model"] == "baseline"].set_index("language")
    adj = pooled[pooled["model"] == "length_adjusted"].set_index("language")
    for lang in ("ar", "he", "ru"):
        assert adj.loc[lang, "or_median"] > base.loc[lang, "or_median"], lang
    # Hebrew and Russian lose the gap entirely once length is held fixed.
    assert adj.loc["he", "or_median"] > 1.0
    assert adj.loc["ru", "or_median"] > 1.0
    # Arabic retains a (weakened) direct effect.
    assert adj.loc["ar", "or_median"] < 1.0


def test_observed_conditional_rates(counts):
    """P(directive | mentioned): RU tracks EN; AR and HE fall well below."""
    assert (counts.loc["en", "n_directive"], counts.loc["en", "n_mentioned"]) == (19, 23)
    assert (counts.loc["ru", "n_directive"], counts.loc["ru", "n_mentioned"]) == (12, 15)
    assert (counts.loc["ar", "n_directive"], counts.loc["ar", "n_mentioned"]) == (7, 15)
    assert (counts.loc["he", "n_directive"], counts.loc["he", "n_mentioned"]) == (10, 17)
    assert counts.loc["ru", "p_directive_given_mention"] == pytest.approx(0.80, abs=0.01)
    assert counts.loc["en", "p_directive_given_mention"] == pytest.approx(0.826, abs=0.01)


def test_arabic_withholds_the_directive_given_a_mention(conditional):
    """The length-independent core finding: AR HDI excludes 1."""
    r = conditional.loc["ar"]
    assert r["or_median"] == pytest.approx(0.22, abs=0.05)
    assert r["hdi_high"] < 1.0, "Arabic conditional HDI must exclude 1"
    assert r["prob_or_lt_1"] > 0.97


def test_russian_does_not_differ_from_english_given_a_mention(conditional):
    """Russian escalates like English: its gap is coverage, not editorial."""
    r = conditional.loc["ru"]
    assert r["or_median"] > 1.0
    assert r["hdi_low"] < 1.0 < r["hdi_high"], "Russian conditional must straddle 1"
    assert r["prob_or_lt_1"] < 0.5


def test_hebrew_is_reported_as_suggestive_only(conditional):
    """Hebrew points with Arabic but sits on the boundary — do not over-claim."""
    r = conditional.loc["he"]
    assert r["or_median"] == pytest.approx(0.30, abs=0.05)
    assert r["prob_or_lt_1"] == pytest.approx(0.95, abs=0.02)
    # Boundary case: the HDI upper bound is ~0.99, i.e. it only just excludes 1.
    assert 0.9 < r["hdi_high"] < 1.05
