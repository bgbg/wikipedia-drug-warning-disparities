"""Tests for the model-reporting helpers (#79): synthetic posteriors, no MCMC, no network."""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g6pd.model_reporting import (  # noqa: E402
    finite_population_ors,
    hierarchical_effect_draws,
    joint_prob_below_one,
    joint_probabilities,
)

LANG = "C(lang, Treatment('en'))"
LANGS = ["ar", "he", "ru"]
TOPICS = ["g6pd", "qt", "teratogenicity"]


def synthetic_idata(beta, offsets, n=2000, seed=0):
    """Posterior with a known structure: beta[L] + noise, offsets[L, T] + noise (2 chains)."""
    rng = np.random.default_rng(seed)
    shape_b, shape_o = (2, n // 2, 3), (2, n // 2, 3, 3)
    b = np.asarray(beta) + 0.01 * rng.standard_normal(shape_b)
    o = np.asarray(offsets) + 0.01 * rng.standard_normal(shape_o)
    sigma = np.abs(rng.standard_normal(shape_b))
    ds = xr.Dataset(
        {
            LANG: (("chain", "draw", f"{LANG}_dim"), b),
            f"{LANG}|topic": (("chain", "draw", f"{LANG}__expr_dim", "topic__factor_dim"), o),
            f"{LANG}|topic_sigma": (("chain", "draw", f"{LANG}__expr_dim"), sigma),
        },
        coords={
            f"{LANG}_dim": LANGS,
            f"{LANG}__expr_dim": LANGS,
            "topic__factor_dim": TOPICS,
        },
    )
    return az.InferenceData(posterior=ds)


def test_joint_probability_known_cases():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(200_000)
    # perfectly correlated columns: the joint equals the marginal (0.5)
    assert joint_prob_below_one(np.exp(np.column_stack([z, z]))) == pytest.approx(0.5, abs=0.01)
    # independent columns: the joint is the product (0.25)
    w = rng.standard_normal(200_000)
    assert joint_prob_below_one(np.exp(np.column_stack([z, w]))) == pytest.approx(0.25, abs=0.01)
    # opposite columns can never both be below 1
    assert joint_prob_below_one(np.exp(np.column_stack([z, -z]))) == 0.0
    # every draw below the threshold
    assert joint_prob_below_one(np.full((10, 4), 0.5)) == 1.0


def test_joint_probability_threshold_is_strict():
    assert joint_prob_below_one(np.array([[1.0, 0.5]])) == 0.0


def test_effect_draws_shapes_and_cells():
    idata = synthetic_idata([-1.0, -2.0, -3.0], np.zeros((3, 3)))
    d = hierarchical_effect_draws(idata)
    assert d["languages"] == LANGS and d["topics"] == TOPICS
    assert d["pooled"].shape == (2000, 3) and d["cells"].shape == (2000, 3, 3)
    assert np.allclose(d["cells"].mean(axis=0)[:, 0], [-1.0, -2.0, -3.0], atol=0.01)


def test_finite_population_is_mean_of_realised_domain_effects():
    beta = [-1.0, -2.0, -3.0]
    offsets = np.zeros((3, 3))
    offsets[:, 0] = [1.0, 0.0, -1.0]  # g6pd offsets
    offsets[:, 1] = [0.5, 0.0, 0.0]   # qt offsets
    idata = synthetic_idata(beta, offsets)
    res = finite_population_ors(idata).set_index(["language", "estimand"])
    # population level: exp(beta); finite population: exp(beta + mean offset over the three domains)
    mean_offset = offsets.mean(axis=1)
    for i, lang in enumerate(LANGS):
        assert res.loc[(lang, "population_level"), "or_median"] == pytest.approx(np.exp(beta[i]), rel=0.02)
        assert res.loc[(lang, "finite_population"), "or_median"] == pytest.approx(
            np.exp(beta[i] + mean_offset[i]), rel=0.02
        )


def test_joint_probabilities_table():
    # every cell clearly below 1 except Russian QT (log OR +2): the nine-cell joint is ~0
    offsets = np.zeros((3, 3))
    offsets[2, 1] = 5.0  # ru (index 2), qt (index 1): -3 + 5 = +2
    idata = synthetic_idata([-1.0, -2.0, -3.0], offsets)
    tab = joint_probabilities(idata).set_index("event")["probability"]
    assert tab["all three pooled ORs below 1"] == 1.0
    assert tab["all nine language by domain ORs below 1"] == 0.0
    assert tab["the eight language by domain ORs other than Russian QT all below 1"] == 1.0
    assert tab["ru: all three domain ORs below 1"] == 0.0
    assert tab["ar: all three domain ORs below 1"] == 1.0
