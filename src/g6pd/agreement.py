"""Agreement statistics for ordinal 0/1/2 warning codes, and the pre-specified validation rule (#77).

The thresholds below are the rule fixed on 3/10/2026, before any collaborator mark was analysed
(methodological_decisions.md). Change them only with a new dated entry there.
"""

import numpy as np
from scipy.stats import binomtest

# Adequacy of the human vs final-code agreement, applied to the per-language point estimate.
KAPPA_ADEQUATE = 0.70
KAPPA_STRONG = 0.80
# Asymmetry criterion: a language is "asymmetric under-detecting" when the exact two-sided sign test
# on the disagreements gives p < ASYMMETRY_ALPHA with the LLM final code BELOW the human code.
ASYMMETRY_ALPHA = 0.10
ASYMMETRY_REFERENCE = "en"        # outcome C requires the asymmetry to be absent here
ASYMMETRY_FOCUS = ("ar", "he")    # outcome C requires it in at least one of these
MIN_STRATUM_N = 8                 # strata with fewer marked cells are not interpretable
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 23
LANGS = ["en", "he", "ar", "ru"]


def weighted_kappa(a, b, labels=(0, 1, 2)) -> float:
    """Cohen's kappa with quadratic weights; NaN when undefined (no variation or no pairs)."""
    a, b = list(a), list(b)
    if not a or len(a) != len(b):
        return float("nan")
    k = len(labels)
    pos = {lab: i for i, lab in enumerate(labels)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[pos[x], pos[y]] += 1
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    w = (np.subtract.outer(np.arange(k), np.arange(k)) / (k - 1)) ** 2
    den = (w * exp).sum()
    return float("nan") if den == 0 else float(1 - (w * obs).sum() / den)


def bootstrap_kappa_ci(a, b, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, level=0.95):
    """Percentile CI of weighted_kappa, resampling cells (pairs) with a fixed seed.

    Draws where kappa is undefined (a resample with no variation) are dropped; the CI is NaN when
    fewer than half of the draws are defined, so a degenerate stratum never gets a spurious interval.
    """
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    ks = np.array([weighted_kappa(a[i], b[i]) for i in idx])
    ks = ks[~np.isnan(ks)]
    if len(ks) < draws / 2:
        return float("nan"), float("nan")
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(ks, [tail, 100 - tail])
    return float(lo), float(hi)


def kappa_band(kappa: float, n: int) -> str:
    """'no data' | 'undefined' | 'strong' | 'adequate' | 'below' (point estimate vs the benchmarks)."""
    if n == 0:
        return "no data"
    if np.isnan(kappa):
        return "undefined"
    return "strong" if kappa >= KAPPA_STRONG else "adequate" if kappa >= KAPPA_ADEQUATE else "below"


def clopper_pearson(k: int, n: int, level: float = 0.95):
    """Exact binomial CI for k successes in n; (NaN, NaN) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method="exact")
    return float(ci.low), float(ci.high)


def sens_spec(truth, pred) -> dict:
    """Sensitivity and specificity of boolean `pred` against boolean `truth`, with exact CIs.

    Empty strata give NaN (and n = 0), never an error.
    """
    truth, pred = np.asarray(truth, dtype=bool), np.asarray(pred, dtype=bool)
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    tp, tn = int((truth & pred).sum()), int((~truth & ~pred).sum())
    out = {"n_pos": n_pos, "n_neg": n_neg}
    for name, k, n in (("sensitivity", tp, n_pos), ("specificity", tn, n_neg)):
        out[name] = k / n if n else float("nan")
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = clopper_pearson(k, n)
    return out


def sign_test_p(n_under: int, n_over: int) -> float:
    """Exact two-sided sign test on the disagreements (binomial, p = 0.5); NaN with none."""
    n = n_under + n_over
    return float("nan") if n == 0 else float(binomtest(min(n_under, n_over), n, 0.5).pvalue)


def asymmetry(n_under: int, n_over: int, n_marked: int = 1) -> str:
    """'no data' | 'under' | 'over' | 'none' for one language.

    'under' is the pre-specified criterion: sign-test p < ASYMMETRY_ALPHA with the LLM below the
    human (n_under counts cells where the LLM code is lower). 'over' is the mirror image, reported
    for information only; the outcome rule uses 'under'.
    """
    if n_marked == 0:
        return "no data"
    p = sign_test_p(n_under, n_over)
    if np.isnan(p) or p >= ASYMMETRY_ALPHA:
        return "none"
    return "under" if n_under > n_over else "over"


def classify_outcome(table) -> tuple[str, str]:
    """Which pre-specified outcome applies, from a per-language table.

    `table` needs columns language, n_cells, weighted_kappa, asymmetry (one row per language, "all"
    ignored). Languages with n_cells == 0 are 'no data', never counted as passing.
      C  asymmetric under-detection in Arabic or Hebrew and not in English (needs English data)
      A  kappa >= KAPPA_ADEQUATE in every language with data, and no asymmetric under-detection
      B  some language below the benchmark, no asymmetric under-detection
    Anything else (asymmetry only in Russian, or also in English; English without data when Arabic
    or Hebrew is asymmetric; kappa undefined) is reported as 'none of A/B/C', and no marks at all as
    'no data', rather than forced into an outcome. The benchmark is applied to the unrounded kappa.
    """
    t = table[table["language"].isin(LANGS)]
    have = t[t["n_cells"] > 0].set_index("language")
    missing = [lang for lang in LANGS if lang not in have.index]
    note = f"languages with data: {', '.join(have.index) or 'none'}; no data: {', '.join(missing) or 'none'}"
    if have.empty:
        return "no data", note
    under = {lang for lang in have.index if have.loc[lang, "asymmetry"] == "under"}
    if under & set(ASYMMETRY_FOCUS):
        if ASYMMETRY_REFERENCE in missing:  # "not in English" cannot be established without English data
            return "none of A/B/C", (note + f"; asymmetric under-detection in {', '.join(sorted(under))}, "
                                     f"but {ASYMMETRY_REFERENCE} has no data, so C cannot be established")
        if ASYMMETRY_REFERENCE not in under:
            return "C", note + f"; asymmetric under-detection in {', '.join(sorted(under))}"
    if under:
        return "none of A/B/C", note + f"; asymmetric under-detection in {', '.join(sorted(under))}, not the pre-specified pattern (ar or he, not en)"
    kappas = have["weighted_kappa"]
    if (kappas < KAPPA_ADEQUATE).any():
        return "B", note + "; below benchmark: " + ", ".join(kappas.index[kappas < KAPPA_ADEQUATE])
    if kappas.isna().any():
        return "none of A/B/C", note + "; kappa undefined in " + ", ".join(kappas.index[kappas.isna()])
    return "A", note
