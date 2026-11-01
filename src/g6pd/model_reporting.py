"""Reporting additions to the hierarchical primary model (#79).

Everything here is derived from the posterior draws of the prespecified model
(`ordinal_analysis.HIER_FORMULA`, unchanged) or from clearly labelled refits:

* ``finite_population_ors``: average of the realised per-domain effects, beside the
  population-level pooled OR.
* ``joint_prob_below_one`` and ``joint_probabilities``: joint posterior probabilities
  that several odds ratios are all below 1.
* ``fit_hierarchical_tau_prior``: the primary model refitted with a different
  HalfNormal prior on the across-domain SD of the language effect (sensitivity).
* ``fit_common_effect`` and ``common_effect_ors``: the POST HOC common-effect model
  ``code ~ 1 + language + topic + (1 | drug)`` (one language effect shared by all domains).

There is no preregistration; the post hoc model is labelled wherever it is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from g6pd.ordinal_analysis import (
    HIER_FORMULA,
    HIER_LANG_TERM,
    HierarchicalFitConfig,
    _hier_terms,
    _or_summary,
    filter_main_fit,
)

COMMON_EFFECT_FORMULA = f"code ~ 1 + {HIER_LANG_TERM} + C(topic) + (1 | drug)"
TAU_TERM = f"{HIER_LANG_TERM}|topic"


def hierarchical_effect_draws(idata) -> dict:
    """Log-OR draws of the hierarchical fit.

    Returns ``{"languages", "topics", "pooled", "cells"}`` where ``pooled`` has shape
    (n_samples, n_languages), the population-level log OR versus English, and ``cells``
    has shape (n_samples, n_languages, n_topics), the realised per-domain log OR
    (population-level effect plus the domain offset), as in ``hierarchical_per_topic_ors``.
    """
    post = idata.posterior
    L, G, _ = _hier_terms(post)
    ld = [d for d in post[L].dims if d not in ("chain", "draw")][0]
    ed = [d for d in post[G].dims if d not in ("chain", "draw") and "topic" not in d][0]
    td = [d for d in post[G].dims if "topic" in d][0]
    langs = [str(x) for x in post[L].coords[ld].values]
    if langs != [str(x) for x in post[G].coords[ed].values]:
        raise ValueError("language order differs between the population and group terms")
    pooled = post[L].stack(s=("chain", "draw")).transpose("s", ld).values
    offsets = post[G].stack(s=("chain", "draw")).transpose("s", ed, td).values
    return {
        "languages": langs,
        "topics": [str(x) for x in post[G].coords[td].values],
        "pooled": pooled,
        "cells": pooled[:, :, None] + offsets,
    }


def finite_population_ors(idata, hdi_prob: float = 0.95) -> pd.DataFrame:
    """Population-level pooled OR beside the finite-population OR, per language.

    population_level:  exp(beta[L])
    finite_population: exp(mean over the realised domains T of beta[L] + offset[L, T])
    """
    d = hierarchical_effect_draws(idata)
    rows = []
    for i, lang in enumerate(d["languages"]):
        estimates = {
            "population_level": d["pooled"][:, i],
            "finite_population": d["cells"][:, i, :].mean(axis=1),
        }
        for name, log_or in estimates.items():
            rows.append({"language": lang, "estimand": name, **_or_summary(np.exp(log_or), hdi_prob)})
    return pd.DataFrame(rows)


def joint_prob_below_one(or_draws: np.ndarray, threshold: float = 1.0) -> float:
    """Posterior probability that every column of ``or_draws`` is below ``threshold`` in the same draw.

    ``or_draws`` has shape (n_samples, k): joint draws of k odds ratios.
    """
    x = np.asarray(or_draws, dtype=float)
    x = x.reshape(len(x), -1)
    return float(np.mean((x < threshold).all(axis=1)))


def joint_probabilities(idata) -> pd.DataFrame:
    """Joint probabilities that sets of odds ratios versus English are all below 1."""
    d = hierarchical_effect_draws(idata)
    langs, topics = d["languages"], d["topics"]
    pooled, cells = np.exp(d["pooled"]), np.exp(d["cells"])
    rows = [
        ("all three pooled ORs below 1", joint_prob_below_one(pooled)),
        ("all nine language by domain ORs below 1", joint_prob_below_one(cells)),
    ]
    keep = np.ones(cells.shape[1:], dtype=bool)
    if "ru" in langs and "qt" in topics:
        keep[langs.index("ru"), topics.index("qt")] = False
        rows.append((
            "the eight language by domain ORs other than Russian QT all below 1",
            joint_prob_below_one(cells[:, keep]),
        ))
    for i, lang in enumerate(langs):
        rows.append((f"{lang}: all three domain ORs below 1", joint_prob_below_one(cells[:, i, :])))
    return pd.DataFrame(rows, columns=["event", "probability"])


def fit_hierarchical_tau_prior(df: pd.DataFrame, tau_sd: float, config: HierarchicalFitConfig | None = None):
    """Refit the primary hierarchical model with ``HalfNormal(tau_sd)`` on the SD of the language effect.

    Everything else (formula, family, ``Normal(0, prior_sd)`` on the language term, seed, sampler
    settings) is as in ``fit_hierarchical_ordinal``. Returns (bambi.Model, arviz.InferenceData).
    """
    import bambi as bmb

    cfg = config or HierarchicalFitConfig()
    work = filter_main_fit(df).copy()
    work["code"] = pd.Categorical(work["code"], categories=[0, 1, 2], ordered=True)
    priors = {
        HIER_LANG_TERM: bmb.Prior("Normal", mu=0, sigma=cfg.prior_sd),
        TAU_TERM: bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=tau_sd)),
    }
    model = bmb.Model(HIER_FORMULA, work, family="cumulative", link="logit", priors=priors)
    idata = model.fit(
        draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, target_accept=cfg.target_accept,
        random_seed=cfg.seed, progressbar=False, idata_kwargs={"log_likelihood": True},
    )
    return model, idata


def fit_common_effect(df: pd.DataFrame, config: HierarchicalFitConfig | None = None):
    """POST HOC: fit ``code ~ 1 + language + topic + (1 | drug)`` (cumulative logit).

    One language effect shared by all domains (domains enter as fixed intercept shifts), the same
    ``Normal(0, prior_sd)`` prior on the language term and the same sampler settings and seed as
    the primary model. Returns (bambi.Model, arviz.InferenceData).
    """
    import bambi as bmb

    cfg = config or HierarchicalFitConfig()
    work = filter_main_fit(df).copy()
    work["code"] = pd.Categorical(work["code"], categories=[0, 1, 2], ordered=True)
    priors = {HIER_LANG_TERM: bmb.Prior("Normal", mu=0, sigma=cfg.prior_sd)}
    model = bmb.Model(COMMON_EFFECT_FORMULA, work, family="cumulative", link="logit", priors=priors)
    idata = model.fit(
        draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, target_accept=cfg.target_accept,
        random_seed=cfg.seed, progressbar=False, idata_kwargs={"log_likelihood": True},
    )
    return model, idata


def common_effect_ors(idata, hdi_prob: float = 0.95) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-language OR versus English from the common-effect fit, plus the joint draws (n_samples, n_languages)."""
    post = idata.posterior
    term = HIER_LANG_TERM
    dim = [d for d in post[term].dims if d not in ("chain", "draw")][0]
    langs = [str(x) for x in post[term].coords[dim].values]
    log_or = post[term].stack(s=("chain", "draw")).transpose("s", dim).values
    rows = [{"language": lang, **_or_summary(np.exp(log_or[:, i]), hdi_prob)} for i, lang in enumerate(langs)]
    return pd.DataFrame(rows), np.exp(log_or)
