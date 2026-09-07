"""Ordinal-aware statistical analysis for cross-lingual warning coverage.

Implements the analysis pipeline for issue #23: Bayesian cumulative-logit
with drug random effect (primary), three robustness checks (frequentist
OrderedModel + cluster-robust SEs, Bayesian prior sensitivity, paired
Wilcoxon), and Cliff's delta as the clinician-facing effect size.

References:
- Capretto et al. (2022). Bambi: A Simple Interface for Fitting Bayesian
  Linear Models in Python. JSS 103(15). https://doi.org/10.18637/jss.v103.i15
- Bürkner & Vuorre (2019). Ordinal Regression Models in Psychology: A
  Tutorial. AMPPS 2(1). https://doi.org/10.1177/2515245918823199
- Betancourt, M. Ordinal Regression (induced-Dirichlet prior).
  https://betanalpha.github.io/assets/case_studies/ordinal_regression.html
- Cliff, N. (1993). Dominance statistics. Psychological Bulletin 114(3),
  494-509.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LANGS = ("en", "he", "ar", "ru")
# PGx-SCAR dropped as an analysed topic 2026-07-01 (see methodological_decisions.md).
TOPICS = ("g6pd", "teratogenicity", "qt")
REFERENCE_LANG = "en"
NON_REFERENCE_LANGS = ("he", "ar", "ru")

# Topic -> the drugs.csv risk column marking a drug as a positive for that topic.
# A drug is a positive for topic T iff its T-risk column is non-empty. Negative
# controls are identified by the `category` column (see ``filter_controls``); a
# clean control additionally has all of the original four risk columns empty
# (see ``config.CONTROL_RISK_COLS``), so the dropped PGx-only drugs are excluded.
TOPIC_RISK_COL = {
    "g6pd": "cpic_risk",
    "teratogenicity": "pregnancy_risk",
    "qt": "qt_risk",
}


def load_coded(coded_csv: Path | str, drugs_csv: Path | str) -> pd.DataFrame:
    """Load coded classifications and join with the drug catalogue.

    Returns a long-form DataFrame with columns:
        drug, lang, topic, code, category

    where `category` is one of {'g6pd_only', 'teratogen_only', 'control'}
    from drugs.csv. The `code` column is the human-adjudicated final code
    when present, otherwise the consensus LLM code; values are int in
    {0, 1, 2}.
    """
    coded = pd.read_csv(coded_csv)
    risk_cols = list(dict.fromkeys(TOPIC_RISK_COL.values()))
    drugs_full = pd.read_csv(drugs_csv).fillna("")
    for col in risk_cols:
        if col not in drugs_full.columns:
            drugs_full[col] = ""
    drugs = drugs_full[["drug", "category", *risk_cols]]

    final = coded["human_final_code"].where(
        coded["human_final_code"].notna(), coded["final_code"]
    )
    out = pd.DataFrame(
        {
            "drug": coded["drug"],
            "lang": coded["lang"],
            "topic": coded["topic"],
            "code": final.astype(int),
        }
    ).merge(drugs, on="drug", how="left")

    missing_category = out["category"].isna()
    if missing_category.any():
        unknown = sorted(out.loc[missing_category, "drug"].unique())
        raise ValueError(f"drugs in coded data missing from drugs.csv: {unknown}")

    return out


def filter_main_fit(df: pd.DataFrame) -> pd.DataFrame:
    """Subset to study-drug (positive) rows for the main cumulative-logit fit.

    Keeps each row where the drug is a *positive* for that row's topic, i.e. the
    topic's risk column (see ``TOPIC_RISK_COL``) is non-empty. This generalises
    the original two-topic, category-based rule to all analysed topics and
    naturally handles multi-topic drugs (phenytoin was a teratogen + PGx-SCAR
    positive; with PGx-SCAR dropped 2026-07-01 it is kept as a teratogen).
    Controls and non-matching (drug, topic) combinations are excluded.
    """
    keep = pd.Series(False, index=df.index)
    for topic, risk_col in TOPIC_RISK_COL.items():
        if risk_col not in df.columns:
            continue
        positive = df[risk_col].astype(str).str.strip() != ""
        keep |= positive & (df["topic"] == topic)
    return df.loc[keep].reset_index(drop=True)


def filter_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Subset to negative-control rows for coding-pipeline validation."""
    return df.loc[df["category"] == "control"].reset_index(drop=True)


def filter_topic(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """Subset to study-drug rows for one topic.

    Applies `filter_main_fit` first (so controls and cross-topic leftover rows
    are dropped), then keeps only the rows for the requested topic. Use this
    for per-topic effect sizes that should be computed on the same subset
    that drives the Bayesian fit.
    """
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}; expected one of {TOPICS}")
    return filter_main_fit(df).pipe(lambda d: d.loc[d["topic"] == topic]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Bayesian cumulative-logit
# -----------------------------------------------------------------------------


@dataclass
class BayesianFitConfig:
    """Configuration for the Bayesian cumulative-logit fit."""

    prior_sd: float = 1.5
    drug_sigma_sd: float = 1.0
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.95
    seed: int = 23
    include_drug_random_effect: bool = True


def prepare_for_bayesian_fit(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns and dtypes the bambi cumulative-logit fit expects.

    - `code` becomes an ordered categorical with levels [0, 1, 2].
    - `cell` is the combined (lang, topic) factor. We use a single combined
      factor (rather than `lang:topic` interaction expansion) because formulae
      drops one level under either parameterisation — by combining them we keep
      a single, easy-to-extract effect array.
    """
    out = df.copy()
    out["code"] = pd.Categorical(out["code"], categories=[0, 1, 2], ordered=True)
    out["cell"] = out["lang"].astype(str) + ":" + out["topic"].astype(str)
    return out


def fit_bayesian_ordinal(df: pd.DataFrame, config: BayesianFitConfig | None = None):
    """Fit the Bayesian cumulative-logit model with a drug random effect.

    Model:
        code ~ 0 + cell + (1 | drug)
        cell  = combined (lang, topic) factor
        family = cumulative, link = logit

    The `cell` factor has 8 levels (4 langs x 2 topics) but formulae drops one
    (alphabetically first: 'ar:g6pd') for identifiability against the
    cumulative-link thresholds. The dropped level acts as the implicit
    baseline with effect 0; posterior summaries below reconstruct the full
    8-cell array before computing contrasts vs. EN.

    Returns the fitted bambi.Model and an arviz.InferenceData object.
    Also calls `model.predict(idata, kind='response_params', inplace=True)`
    so the posterior includes per-row P(code = k) used for posterior
    predictive composition.
    """
    import bambi as bmb

    cfg = config or BayesianFitConfig()
    work = prepare_for_bayesian_fit(df)

    if cfg.include_drug_random_effect:
        formula = "code ~ 0 + cell + (1 | drug)"
    else:
        formula = "code ~ 0 + cell"

    priors = {
        "cell": bmb.Prior("Normal", mu=0, sigma=cfg.prior_sd),
    }
    if cfg.include_drug_random_effect:
        priors["1|drug"] = bmb.Prior(
            "Normal",
            mu=0,
            sigma=bmb.Prior("HalfNormal", sigma=cfg.drug_sigma_sd),
        )

    model = bmb.Model(
        formula,
        work,
        family="cumulative",
        link="logit",
        priors=priors,
    )
    idata = model.fit(
        draws=cfg.draws,
        tune=cfg.tune,
        chains=cfg.chains,
        target_accept=cfg.target_accept,
        random_seed=cfg.seed,
        progressbar=False,
        idata_kwargs={"log_likelihood": True},
    )
    model.predict(idata, kind="response_params", inplace=True)
    return model, idata


# -----------------------------------------------------------------------------
# Hierarchical (partial-pooling) cumulative-logit — the four-topic headline
# -----------------------------------------------------------------------------

# EN is forced as the language reference via Treatment('en') (otherwise formulae
# drops 'ar' alphabetically). The varying term (language | topic) partially pools
# the language effect across the four topics, so the model yields one pooled
# non-English-vs-EN odds ratio per language plus tau, the posterior SD of the
# per-topic language effects — a formal measure of how *systematic* the gap is.
HIER_LANG_TERM = "C(lang, Treatment('en'))"
HIER_FORMULA = (
    f"code ~ 1 + {HIER_LANG_TERM} + (1 | drug) + ({HIER_LANG_TERM} | topic)"
)


@dataclass
class HierarchicalFitConfig:
    """Configuration for the hierarchical cumulative-logit fit."""

    prior_sd: float = 1.5
    draws: int = 1500
    tune: int = 1500
    chains: int = 4
    target_accept: float = 0.98
    seed: int = 23


def fit_hierarchical_ordinal(df: pd.DataFrame, config: HierarchicalFitConfig | None = None):
    """Fit the hierarchical cumulative-logit model on the main-fit (positives) subset.

    Model:
        code ~ 1 + C(lang, Treatment('en')) + (1 | drug)
                 + (C(lang, Treatment('en')) | topic)
        family = cumulative, link = logit

    Returns (bambi.Model, arviz.InferenceData). The caller extracts pooled ORs,
    per-topic ORs, and tau with the helpers below.
    """
    import bambi as bmb

    cfg = config or HierarchicalFitConfig()
    work = filter_main_fit(df).copy()
    work["code"] = pd.Categorical(work["code"], categories=[0, 1, 2], ordered=True)

    priors = {HIER_LANG_TERM: bmb.Prior("Normal", mu=0, sigma=cfg.prior_sd)}
    model = bmb.Model(HIER_FORMULA, work, family="cumulative", link="logit", priors=priors)
    idata = model.fit(
        draws=cfg.draws,
        tune=cfg.tune,
        chains=cfg.chains,
        target_accept=cfg.target_accept,
        random_seed=cfg.seed,
        progressbar=False,
        idata_kwargs={"log_likelihood": True},
    )
    return model, idata


def _hier_terms(post):
    """Return (pop_term, group_term, sigma_term) variable names from the posterior."""
    lang = [v for v in post.data_vars if v.startswith("C(lang") and "sigma" not in v and "|" not in v]
    group = [v for v in post.data_vars if v.startswith("C(lang") and "|" in v and "sigma" not in v]
    sigma = [v for v in post.data_vars if v.startswith("C(lang") and "sigma" in v]
    if not (lang and group and sigma):
        raise KeyError("hierarchical posterior missing expected C(lang...) terms")
    return lang[0], group[0], sigma[0]


def _or_summary(or_vals: np.ndarray, hdi_prob: float) -> dict:
    """Median-centred OR summary (median is robust to the heavy exp tails)."""
    import arviz as az

    hdi = az.hdi(or_vals, hdi_prob=hdi_prob)
    return {
        "or_median": float(np.median(or_vals)),
        "or_mean": float(np.mean(or_vals)),
        "hdi_low": float(hdi[0]),
        "hdi_high": float(hdi[1]),
        "prob_or_lt_1": float(np.mean(or_vals < 1)),
    }


def or_summary(or_vals: np.ndarray, hdi_prob: float = 0.95) -> dict:
    """Public wrapper over ``_or_summary`` for callers outside this module.

    Scripts that summarise their own posterior draws (e.g.
    ``scripts/length_sensitivity.py``, whose conditional model has no
    ``(lang | topic)`` group term and so cannot use ``hierarchical_pooled_ors``)
    need the same median + az.hdi convention used for the §5b tables. Exposing
    it here keeps that convention in one place instead of having callers reach
    for the private name.
    """
    return _or_summary(or_vals, hdi_prob)


def hierarchical_pooled_ors(idata, hdi_prob: float = 0.95) -> pd.DataFrame:
    """Pooled (partial-pooled across topics) cumulative OR vs EN, per language.

    OR(L) = exp(beta_lang[L]); prob_or_lt_1 is the posterior probability that the
    non-English edition carries a *weaker* warning than English, pooled over the
    four topics.
    """
    post = idata.posterior
    L, _, _ = _hier_terms(post)
    ld = [d for d in post[L].dims if d not in ("chain", "draw")][0]
    rows = []
    for i, lang in enumerate(post[L].coords[ld].values):
        or_vals = np.exp(post[L].isel({ld: i}).stack(s=("chain", "draw")).values)
        rows.append({"language": str(lang), **_or_summary(or_vals, hdi_prob)})
    return pd.DataFrame(rows)


def hierarchical_per_topic_ors(idata, hdi_prob: float = 0.95) -> pd.DataFrame:
    """Per-(language, topic) cumulative OR vs EN: exp(beta_lang[L] + offset[L, T])."""
    post = idata.posterior
    L, G, _ = _hier_terms(post)
    ld = [d for d in post[L].dims if d not in ("chain", "draw")][0]
    ed = [d for d in post[G].dims if d not in ("chain", "draw") and "topic" not in d][0]
    td = [d for d in post[G].dims if "topic" in d][0]
    rows = []
    for i, lang in enumerate(post[L].coords[ld].values):
        for j, topic in enumerate(post[G].coords[td].values):
            beta = (
                post[L].isel({ld: i}).stack(s=("chain", "draw")).values
                + post[G].isel({ed: i, td: j}).stack(s=("chain", "draw")).values
            )
            rows.append(
                {"language": str(lang), "topic": str(topic), **_or_summary(np.exp(beta), hdi_prob)}
            )
    return pd.DataFrame(rows)


def hierarchical_tau(idata, hdi_prob: float = 0.95) -> pd.DataFrame:
    """Posterior of tau = SD of the per-topic language effect, per language.

    Small tau (relative to the pooled effect) means the language gap is
    consistent across topics — the quantitative form of "systematic".
    """
    import arviz as az

    post = idata.posterior
    _, _, S = _hier_terms(post)
    sd = [d for d in post[S].dims if d not in ("chain", "draw")][0]
    rows = []
    for i, lang in enumerate(post[S].coords[sd].values):
        vals = post[S].isel({sd: i}).stack(s=("chain", "draw")).values
        hdi = az.hdi(vals, hdi_prob=hdi_prob)
        rows.append({
            "language": str(lang),
            "tau_median": float(np.median(vals)),
            "tau_mean": float(np.mean(vals)),
            "hdi_low": float(hdi[0]),
            "hdi_high": float(hdi[1]),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Posterior summaries
# -----------------------------------------------------------------------------


def _stack_chain_draw(arr):
    """Stack chain and draw dims into a single 'sample' dim."""
    return arr.stack(sample=("chain", "draw"))


def _lang_topic_effects(idata) -> "xr.DataArray":
    """Extract per-cell linear-predictor effects from the posterior.

    Reads the `cell` term (combined lang:topic factor) and reconstructs a
    full (sample, language, topic) array. The level that formulae drops for
    identifiability is filled in with zeros (it is the implicit baseline).

    Returns DataArray with dims (sample, language, topic).
    """
    import xarray as xr

    posterior = idata.posterior
    if "cell" not in posterior:
        raise KeyError(
            "expected 'cell' variable in posterior; did you fit with "
            "the `0 + cell + (1|drug)` parameterisation?"
        )

    effect = posterior["cell"]
    cell_dim = next(d for d in effect.dims if d not in ("chain", "draw"))
    labels = [str(x) for x in effect.coords[cell_dim].values]

    stacked = _stack_chain_draw(effect)
    samples = stacked.values.T  # (n_samples, n_estimated_cells)
    n_samples = samples.shape[0]

    out = np.zeros((n_samples, len(LANGS), len(TOPICS)))  # baseline = 0
    seen = set()
    for j, label in enumerate(labels):
        lg, tp = label.split(":")
        out[:, LANGS.index(lg), TOPICS.index(tp)] = samples[:, j]
        seen.add((lg, tp))

    expected = {(lg, tp) for lg in LANGS for tp in TOPICS}
    missing = expected - seen
    if len(missing) != 1:
        raise RuntimeError(
            f"expected exactly one dropped baseline cell, got {missing}"
        )
    expected_baseline = ("ar", "g6pd")
    if expected_baseline not in missing:
        actual = next(iter(missing))
        raise RuntimeError(
            f"formulae dropped a different baseline cell than expected: "
            f"got {actual}, expected {expected_baseline}. The docstring of "
            f"`fit_bayesian_ordinal` and downstream interpretations assume "
            f"`ar:g6pd` is the implicit zero baseline. If formulae's "
            f"behaviour has changed, update both the docstring and any "
            f"interpretation code that names the baseline explicitly."
        )

    return xr.DataArray(
        out,
        dims=("sample", "language", "topic"),
        coords={"language": list(LANGS), "topic": list(TOPICS)},
    )


def cumulative_or_vs_reference(
    idata, reference: str = REFERENCE_LANG, hdi_prob: float = 0.95
) -> pd.DataFrame:
    """Cumulative odds ratios for each (non-reference language, topic) vs. reference.

    OR(L, T) = exp(beta[L, T] - beta[reference, T]).

    Returns DataFrame with columns:
        language, topic, mean, sd, hdi_low, hdi_high, prob_or_gt_1
    """
    import arviz as az

    effects = _lang_topic_effects(idata)
    beta_ref = effects.sel(language=reference)
    rows = []
    for lg in [l for l in LANGS if l != reference]:
        for tp in TOPICS:
            diff = effects.sel(language=lg, topic=tp) - beta_ref.sel(topic=tp)
            log_or = diff.values
            or_vals = np.exp(log_or)
            hdi = az.hdi(or_vals, hdi_prob=hdi_prob)
            rows.append(
                {
                    "language": lg,
                    "topic": tp,
                    "mean": float(np.mean(or_vals)),
                    "sd": float(np.std(or_vals)),
                    "hdi_low": float(hdi[0]),
                    "hdi_high": float(hdi[1]),
                    "prob_or_gt_1": float(np.mean(or_vals > 1)),
                }
            )
    return pd.DataFrame(rows)


def prob_stochastic_dominance(
    idata, reference: str = REFERENCE_LANG
) -> pd.DataFrame:
    """Posterior probability that reference language stochastically dominates.

    Defined as P(beta[reference, topic] > beta[L, topic] | data), which is
    the posterior probability that an article in the reference language
    has a higher latent warning level than one in language L.

    Returns DataFrame with columns: language, topic, prob_dominance.
    """
    effects = _lang_topic_effects(idata)
    beta_ref = effects.sel(language=reference)
    rows = []
    for lg in [l for l in LANGS if l != reference]:
        for tp in TOPICS:
            diff = beta_ref.sel(topic=tp).values - effects.sel(language=lg, topic=tp).values
            rows.append(
                {
                    "language": lg,
                    "topic": tp,
                    "prob_dominance": float(np.mean(diff > 0)),
                }
            )
    return pd.DataFrame(rows)


def interaction_contrast(idata, reference: str = REFERENCE_LANG, hdi_prob: float = 0.95) -> pd.DataFrame:
    """Difference-in-differences: does the cross-lingual gap differ by topic?

    For each non-reference language L, computes the posterior of
        (beta[L, terato] - beta[ref, terato]) - (beta[L, g6pd] - beta[ref, g6pd])

    A non-zero posterior here indicates the gap differs by topic.
    Returns DataFrame with columns:
        language, mean, sd, hdi_low, hdi_high, prob_positive.
    """
    import arviz as az

    effects = _lang_topic_effects(idata)
    rows = []
    for lg in [l for l in LANGS if l != reference]:
        gap_terato = (
            effects.sel(language=lg, topic="teratogenicity")
            - effects.sel(language=reference, topic="teratogenicity")
        ).values
        gap_g6pd = (
            effects.sel(language=lg, topic="g6pd")
            - effects.sel(language=reference, topic="g6pd")
        ).values
        diff = gap_terato - gap_g6pd
        hdi = az.hdi(diff, hdi_prob=hdi_prob)
        rows.append(
            {
                "language": lg,
                "mean": float(np.mean(diff)),
                "sd": float(np.std(diff)),
                "hdi_low": float(hdi[0]),
                "hdi_high": float(hdi[1]),
                "prob_positive": float(np.mean(diff > 0)),
            }
        )
    return pd.DataFrame(rows)


def rope_summary(
    idata,
    rope_or: tuple[float, float] = (0.5, 2.0),
    reference: str = REFERENCE_LANG,
) -> pd.DataFrame:
    """Posterior mass of cumulative ORs inside vs. outside a ROPE on the OR scale.

    Returns DataFrame with columns:
        language, topic, rope_low, rope_high, prob_in_rope, prob_or_lt_low,
        prob_or_gt_high.
    """
    rope_low, rope_high = rope_or
    effects = _lang_topic_effects(idata)
    beta_ref = effects.sel(language=reference)
    rows = []
    for lg in [l for l in LANGS if l != reference]:
        for tp in TOPICS:
            diff = effects.sel(language=lg, topic=tp).values - beta_ref.sel(topic=tp).values
            or_vals = np.exp(diff)
            rows.append(
                {
                    "language": lg,
                    "topic": tp,
                    "rope_low": rope_low,
                    "rope_high": rope_high,
                    "prob_in_rope": float(np.mean((or_vals >= rope_low) & (or_vals <= rope_high))),
                    "prob_or_lt_low": float(np.mean(or_vals < rope_low)),
                    "prob_or_gt_high": float(np.mean(or_vals > rope_high)),
                }
            )
    return pd.DataFrame(rows)


def posterior_predictive_composition(
    idata, df: pd.DataFrame, hdi_prob: float = 0.95
) -> pd.DataFrame:
    """Posterior predictive {0, 1, 2} composition per (language, topic).

    Uses the response_params (P(code = k | row)) emitted by bambi.predict
    and averages across all rows in each (language, topic) cell.

    Returns DataFrame with columns:
        language, topic, code, mean, hdi_low, hdi_high, n_obs.
    """
    import arviz as az

    posterior = idata.posterior
    if "p" in posterior.data_vars:
        probs = posterior["p"]
    else:
        candidates = [
            n for n, da in posterior.data_vars.items() if da.ndim >= 4
        ]
        if not candidates:
            raise KeyError(
                "response probabilities not found; "
                "did you fit with model.predict(idata, kind='response_params')?"
            )
        probs = posterior[candidates[0]]

    obs_dim = next(
        d
        for d in probs.dims
        if d not in ("chain", "draw") and probs.sizes[d] == len(df)
    )
    code_dim = next(d for d in probs.dims if d not in ("chain", "draw", obs_dim))
    n_codes = probs.sizes[code_dim]

    sample = _stack_chain_draw(probs).transpose("sample", obs_dim, code_dim).values
    # shape: (n_samples, n_obs, n_codes)

    rows = []
    for lg in LANGS:
        for tp in TOPICS:
            mask = (df["lang"].values == lg) & (df["topic"].values == tp)
            n_obs = int(mask.sum())
            if n_obs == 0:
                continue
            cell = sample[:, mask, :].mean(axis=1)  # (n_samples, n_codes)
            for k in range(n_codes):
                hdi = az.hdi(cell[:, k], hdi_prob=hdi_prob)
                rows.append(
                    {
                        "language": lg,
                        "topic": tp,
                        "code": k,
                        "mean": float(cell[:, k].mean()),
                        "hdi_low": float(hdi[0]),
                        "hdi_high": float(hdi[1]),
                        "n_obs": n_obs,
                    }
                )
    return pd.DataFrame(rows)


def observed_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Observed {0, 1, 2} composition per (language, topic) cell."""
    rows = []
    for lg in LANGS:
        for tp in TOPICS:
            cell = df[(df["lang"] == lg) & (df["topic"] == tp)]
            n_obs = len(cell)
            if n_obs == 0:
                continue
            for k in (0, 1, 2):
                rows.append(
                    {
                        "language": lg,
                        "topic": tp,
                        "code": k,
                        "count": int((cell["code"] == k).sum()),
                        "proportion": float((cell["code"] == k).mean()),
                        "n_obs": n_obs,
                    }
                )
    return pd.DataFrame(rows)


def convergence_diagnostics(idata) -> pd.DataFrame:
    """Compute R-hat and bulk ESS for all parameters."""
    import arviz as az

    summary = az.summary(idata, kind="diagnostics")
    return summary.reset_index().rename(columns={"index": "parameter"})


# -----------------------------------------------------------------------------
# Frequentist robustness check (statsmodels OrderedModel + cluster-robust SE)
# -----------------------------------------------------------------------------


def fit_frequentist_ordered(df: pd.DataFrame) -> dict:
    """Fit `statsmodels.OrderedModel` with cluster-robust SEs on drug.

    Returns a dict with keys:
        result      - fitted OrderedResultsWrapper
        ors         - DataFrame of cumulative ORs vs. EN per (language, topic)
                      with cluster-robust 95% CIs (Wald)
        cov_cluster - ndarray (n_params, n_params), cluster-robust covariance
        param_index - Index of parameter names matching cov_cluster
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    work = df.copy()
    work["lang"] = pd.Categorical(work["lang"], categories=LANGS)
    work["topic"] = pd.Categorical(work["topic"], categories=TOPICS)

    formula = (
        "code ~ C(lang, Treatment(reference='en')) * "
        "C(topic, Treatment(reference='g6pd'))"
    )
    model = OrderedModel.from_formula(formula, data=work, distr="logit")
    fit = model.fit(method="bfgs", disp=False)

    cluster_groups = work["drug"].astype("category").cat.codes.values
    cov_cluster = _cluster_robust_cov(fit, cluster_groups)
    param_index = fit.params.index

    ors = _ordered_model_or_table(fit, cov_cluster, param_index)
    return {
        "result": fit,
        "ors": ors,
        "cov_cluster": cov_cluster,
        "param_index": param_index,
    }


def _cluster_robust_cov(fit, groups: np.ndarray) -> np.ndarray:
    """Cluster-robust sandwich variance estimator.

    var = (H^-1) (sum_g s_g s_g') (H^-1)
    where s_g is the cluster-summed score and H^-1 is the inverse Hessian
    (which `cov_params()` returns at the MLE under standard ML estimation).
    Includes the Cameron & Miller (2015) small-sample adjustment.
    """
    score_obs = np.asarray(fit.model.score_obs(fit.params))
    cov_inv_hess = np.asarray(fit.cov_params())

    unique_groups = np.unique(groups)
    n_params = score_obs.shape[1]
    meat = np.zeros((n_params, n_params))
    for g in unique_groups:
        s_g = score_obs[groups == g].sum(axis=0)
        meat += np.outer(s_g, s_g)

    n_groups = unique_groups.size
    n_obs = score_obs.shape[0]
    adj = (n_groups / max(n_groups - 1, 1)) * (
        (n_obs - 1) / max(n_obs - n_params, 1)
    )
    meat *= adj

    return cov_inv_hess @ meat @ cov_inv_hess


def _ordered_model_or_table(
    fit, cov: np.ndarray, param_index: pd.Index
) -> pd.DataFrame:
    """Cumulative ORs vs. EN from OrderedModel coefficients with cluster SE.

    Reference is hardcoded to EN to match the formula's `Treatment(reference='en')`
    contrast — the frequentist robustness check always reports vs. EN.

    Builds a contrast vector `c` for each (language, topic) cell such that
    `c' beta = log_or(L, T)` and propagates the cluster-robust covariance:
        var(log_or) = c' cov c
    """
    params = fit.params
    rows = []
    for lg in [l for l in LANGS if l != REFERENCE_LANG]:
        for tp in TOPICS:
            contrast = np.zeros(len(param_index))
            lang_term = f"C(lang, Treatment(reference='en'))[T.{lg}]"
            if lang_term in param_index:
                contrast[param_index.get_loc(lang_term)] = 1.0
            if tp == "teratogenicity":
                inter_term = (
                    f"C(lang, Treatment(reference='en'))[T.{lg}]:"
                    "C(topic, Treatment(reference='g6pd'))[T.teratogenicity]"
                )
                if inter_term in param_index:
                    contrast[param_index.get_loc(inter_term)] = 1.0
            log_or = float(contrast @ params.values)
            var = float(contrast @ cov @ contrast)
            log_or_se = float(np.sqrt(max(var, 0.0)))
            ci_low = float(np.exp(log_or - 1.96 * log_or_se))
            ci_high = float(np.exp(log_or + 1.96 * log_or_se))
            rows.append(
                {
                    "language": lg,
                    "topic": tp,
                    "or_estimate": float(np.exp(log_or)),
                    "log_or": log_or,
                    "log_or_se": log_or_se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Cliff's delta with cluster bootstrap
# -----------------------------------------------------------------------------


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    """Cliff's delta = P(X > Y) - P(X < Y).

    Returns a value in [-1, 1]. Positive: X tends to be larger than Y.
    """
    x_arr = np.asarray(list(x))
    y_arr = np.asarray(list(y))
    if x_arr.size == 0 or y_arr.size == 0:
        return float("nan")
    diffs = x_arr[:, None] - y_arr[None, :]
    n_gt = float((diffs > 0).sum())
    n_lt = float((diffs < 0).sum())
    return (n_gt - n_lt) / (x_arr.size * y_arr.size)


def cliffs_delta_paired_bootstrap(
    df: pd.DataFrame,
    *,
    lang_a: str,
    lang_b: str,
    topic: str,
    n_boot: int = 2000,
    seed: int = 23,
    ci_prob: float = 0.95,
) -> dict:
    """Cluster-bootstrap CI for Cliff's delta, resampling drugs.

    The data are paired by drug: for each drug we have one code in each
    language. We resample drug labels with replacement and recompute
    Cliff's delta on (lang_a vs. lang_b) within the chosen topic. This
    respects the within-drug correlation that simple row-level bootstrap
    would ignore.
    """
    sub = df[(df["topic"] == topic) & (df["lang"].isin((lang_a, lang_b)))]
    pivot = sub.pivot_table(index="drug", columns="lang", values="code", aggfunc="first")
    pivot = pivot.dropna(subset=[lang_a, lang_b])
    drugs = pivot.index.values
    a_obs = pivot[lang_a].values.astype(float)
    b_obs = pivot[lang_b].values.astype(float)

    point = cliffs_delta(a_obs, b_obs)

    rng = np.random.default_rng(seed)
    n_drugs = len(drugs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_drugs, size=n_drugs)
        boots[i] = cliffs_delta(a_obs[idx], b_obs[idx])

    alpha = (1 - ci_prob) / 2
    ci_low = float(np.quantile(boots, alpha))
    ci_high = float(np.quantile(boots, 1 - alpha))
    return {
        "lang_a": lang_a,
        "lang_b": lang_b,
        "topic": topic,
        "n_drugs": n_drugs,
        "delta": float(point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": n_boot,
    }


def cliffs_delta_table(
    df: pd.DataFrame,
    *,
    reference: str = REFERENCE_LANG,
    n_boot: int = 2000,
    seed: int = 23,
) -> pd.DataFrame:
    """Cliff's delta + cluster-bootstrap CI for each non-reference language vs. reference, per topic.

    Convention: delta = P(X_ref > X_other) - P(X_ref < X_other), so positive
    delta means the reference language warns more strongly than the other.
    """
    rows = []
    for lg in [l for l in LANGS if l != reference]:
        for tp in TOPICS:
            res = cliffs_delta_paired_bootstrap(
                df,
                lang_a=reference,
                lang_b=lg,
                topic=tp,
                n_boot=n_boot,
                seed=seed,
            )
            rows.append(res)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Paired Wilcoxon
# -----------------------------------------------------------------------------


def paired_wilcoxon_table(
    df: pd.DataFrame, *, reference: str = REFERENCE_LANG
) -> pd.DataFrame:
    """Wilcoxon signed-rank test, paired by drug, for each (non-ref language, topic).

    Drops drug pairs where both codes are equal (zero method='wilcox' default).
    """
    from scipy.stats import wilcoxon

    rows = []
    for lg in [l for l in LANGS if l != reference]:
        for tp in TOPICS:
            sub = df[(df["topic"] == tp) & (df["lang"].isin((reference, lg)))]
            pivot = sub.pivot_table(
                index="drug", columns="lang", values="code", aggfunc="first"
            ).dropna(subset=[reference, lg])
            x = pivot[reference].values.astype(float)
            y = pivot[lg].values.astype(float)
            n_pairs = len(x)
            n_nonzero = int((x != y).sum())
            if n_nonzero == 0:
                rows.append(
                    {
                        "language": lg,
                        "topic": tp,
                        "n_pairs": n_pairs,
                        "n_nonzero_diffs": 0,
                        "statistic": float("nan"),
                        "pvalue": float("nan"),
                    }
                )
                continue
            try:
                stat, pval = wilcoxon(
                    x, y, zero_method="wilcox", alternative="two-sided"
                )
                rows.append(
                    {
                        "language": lg,
                        "topic": tp,
                        "n_pairs": n_pairs,
                        "n_nonzero_diffs": n_nonzero,
                        "statistic": float(stat),
                        "pvalue": float(pval),
                    }
                )
            except ValueError as exc:
                rows.append(
                    {
                        "language": lg,
                        "topic": tp,
                        "n_pairs": n_pairs,
                        "n_nonzero_diffs": n_nonzero,
                        "statistic": float("nan"),
                        "pvalue": float("nan"),
                        "note": str(exc),
                    }
                )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Negative-control validation
# -----------------------------------------------------------------------------


def prior_implied_or_samples(
    prior_sd: float, n: int = 10_000, seed: int = 23
) -> np.ndarray:
    """Sample from the implied prior on cumulative ORs vs. reference.

    Two cell effects iid Normal(0, prior_sd) imply the difference is
    Normal(0, prior_sd * sqrt(2)) and the OR is the exponentiated
    difference. Returns `n` samples of the implied OR.
    """
    rng = np.random.default_rng(seed)
    diff = rng.normal(0, prior_sd, n) - rng.normal(0, prior_sd, n)
    return np.exp(diff)


# -----------------------------------------------------------------------------
# Sensitivity analyses
# -----------------------------------------------------------------------------


def prior_sensitivity_table(
    df: pd.DataFrame,
    prior_sds: Iterable[float] = (0.5, 1.5, 5.0),
    *,
    base_config: BayesianFitConfig | None = None,
) -> pd.DataFrame:
    """Refit the Bayesian model at multiple prior_sd values; collect ORs.

    Returns a long DataFrame stacking `cumulative_or_vs_reference` outputs
    annotated with the `prior_sd` used. Caller can pivot/compare.
    """
    base = base_config or BayesianFitConfig()
    out = []
    for sd in prior_sds:
        cfg = BayesianFitConfig(
            prior_sd=sd,
            drug_sigma_sd=base.drug_sigma_sd,
            draws=base.draws,
            tune=base.tune,
            chains=base.chains,
            target_accept=base.target_accept,
            seed=base.seed,
            include_drug_random_effect=base.include_drug_random_effect,
        )
        _, idata = fit_bayesian_ordinal(df, cfg)
        ors = cumulative_or_vs_reference(idata)
        ors["prior_sd"] = float(sd)
        out.append(ors)
    return pd.concat(out, ignore_index=True)


def fit_no_random_effect(
    df: pd.DataFrame, base_config: BayesianFitConfig | None = None
):
    """Convenience: refit the Bayesian model with `(1 | drug)` removed."""
    base = base_config or BayesianFitConfig()
    cfg = BayesianFitConfig(
        prior_sd=base.prior_sd,
        drug_sigma_sd=base.drug_sigma_sd,
        draws=base.draws,
        tune=base.tune,
        chains=base.chains,
        target_accept=base.target_accept,
        seed=base.seed,
        include_drug_random_effect=False,
    )
    return fit_bayesian_ordinal(df, cfg)


DOCUMENTED_CONTROL_EXCEPTIONS: dict[tuple[str, str, str], str] = {
    # See methodological_decisions.md (2026-03-23, Rule 4 / atorvastatin caveat)
    ("atorvastatin", "en", "teratogenicity"): "FDA pregnancy category rescinded 2021; Wikipedia regulatory lag",
    ("atorvastatin", "ar", "teratogenicity"): "FDA pregnancy category rescinded 2021; Wikipedia regulatory lag",
    ("omeprazole", "ar", "teratogenicity"): "Arabic Wikipedia lists pregnancy under contraindications (Rule 4)",
}


def validate_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Report code per (control drug, language, topic).

    Adds two flag columns:
      `deviates` — True when code != 0 (the structural expectation for a
                   negative control).
      `documented_exception` — True when the deviation is pre-explained in
                   `methodological_decisions.md` (see
                   `DOCUMENTED_CONTROL_EXCEPTIONS`). A row where
                   `deviates and not documented_exception` is the only kind
                   that actually requires investigation.
    """
    controls = filter_controls(df)
    out = controls[["drug", "lang", "topic", "code"]].copy()
    out["expected"] = 0
    out["deviates"] = out["code"] != 0
    out["documented_exception"] = out.apply(
        lambda r: (r["drug"], r["lang"], r["topic"]) in DOCUMENTED_CONTROL_EXCEPTIONS,
        axis=1,
    )
    out["exception_note"] = out.apply(
        lambda r: DOCUMENTED_CONTROL_EXCEPTIONS.get((r["drug"], r["lang"], r["topic"]), ""),
        axis=1,
    )
    return out.sort_values(["drug", "topic", "lang"]).reset_index(drop=True)
