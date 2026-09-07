"""
length_sensitivity.py — is the cross-lingual warning gap just an article-length
artefact? (#65)

Answers the reviewer objection that non-English Wikipedia articles are simply
shorter, and that the language effect is therefore a length artefact. Two models:

1. **Length-adjusted hierarchical** — the §5b model refit with `log(word count)`
   as a covariate, on the subset of (drug, language, topic) cells whose article
   exists. A baseline model is refit on the *same rows* so the comparison is
   like-for-like (differences are the covariate, not the subsetting).

2. **Conditional (mention -> directive)** — among cells where the risk is
   mentioned at all (code >= 1), the probability that the article escalates to
   an explicit directive (code == 2), with a per-drug random effect. This model
   is length-independent *by construction*: it conditions on the topic being
   covered, so "shorter articles contain less of everything" cannot explain it.

**Estimand note (important).** Word count is a **mediator**, not a confounder:
edition -> smaller editor community -> shorter article -> weaker warning. The
drug is held fixed by the per-drug random effect, and nothing assigns which
edition an article lives in. Adjusting for a mediator therefore estimates a
*controlled direct effect* — a different quantity from the total effect reported
in §5b — and risks overadjustment and collider bias via unmeasured editorial
investment. The total effect remains the paper's headline; this script's output
is a mechanism analysis, not a correction. Language and length are also
near-collinear, so the length-adjusted language coefficient is weakly
identified: attenuation here is not refutation.

Outputs (under --output-dir):
    pooled_ors_by_model.csv  - pooled OR vs EN, baseline vs length-adjusted
    log_wc_effect.csv        - the log(word count) coefficient as an OR
    conditional_ors.csv      - OR of giving a directive vs EN, given a mention
    conditional_counts.csv   - observed P(code 2 | code >= 1) per edition
    manifest.json            - run metadata

Usage:
    python scripts/length_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g6pd.config import ARTICLES_JSONL, CODED_DIR, DRUGS_CSV  # noqa: E402
from g6pd import ordinal_analysis as oa  # noqa: E402

LANG_TERM = oa.HIER_LANG_TERM
LANG_ORDER = ["ru", "ar", "he"]


def load_word_counts(articles_jsonl: Path) -> dict[tuple[str, str], int]:
    """{(drug, lang): word_count} for articles that exist and carry a count.

    Tests ``word_count is not None`` rather than its truthiness: a genuinely
    zero-length existing article must stay in the subset as length 0, not be
    silently reclassified as "no article" and dropped from the length model.
    """
    wc: dict[tuple[str, str], int] = {}
    with open(articles_jsonl, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("exists") and r.get("word_count") is not None:
                wc[(r["drug"], r["lang"])] = int(r["word_count"])
    return wc


def _fit(formula: str, data: pd.DataFrame, family: str, cfg) -> "object":
    import bambi as bmb

    work = data.copy()
    if family == "cumulative":
        work["code"] = pd.Categorical(work["code"], categories=[0, 1, 2], ordered=True)
    priors = {LANG_TERM: bmb.Prior("Normal", mu=0, sigma=cfg.prior_sd)}
    model = bmb.Model(formula, work, family=family, link="logit", priors=priors)
    return model.fit(
        draws=cfg.draws,
        tune=cfg.tune,
        chains=cfg.chains,
        target_accept=cfg.target_accept,
        random_seed=cfg.seed,
        progressbar=False,
    )


def _lang_or_table(idata, hdi_prob: float = 0.95) -> pd.DataFrame:
    """OR vs EN per language from a posterior carrying the C(lang...) term.

    Summarised with ``oa.or_summary`` (median + az.hdi) so the intervals match
    the convention used for the §5b hierarchical model rather than mixing in
    equal-tailed percentiles, which differ materially on these skewed posteriors.
    Used for the conditional model, which has no (lang | topic) group term and so
    cannot go through ``oa.hierarchical_pooled_ors``.
    """
    post = idata.posterior
    term = [
        v
        for v in post.data_vars
        if v.startswith("C(lang") and "sigma" not in v and "|" not in v
    ][0]
    dim = [d for d in post[term].dims if d not in ("chain", "draw")][0]
    rows = []
    for i, lang in enumerate(post[term].coords[dim].values):
        vals = np.exp(post[term].isel({dim: i}).stack(s=("chain", "draw")).values)
        rows.append({"language": str(lang), **oa.or_summary(vals, hdi_prob)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Length-confound sensitivity analysis (#65)")
    p.add_argument("--input", default=str(CODED_DIR / "coded_four_topic.csv"))
    p.add_argument("--drugs", default=str(DRUGS_CSV))
    p.add_argument("--articles", default=str(ARTICLES_JSONL))
    p.add_argument("--output-dir", default=str(CODED_DIR / "length_sensitivity"))
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--draws", type=int, default=1500)
    p.add_argument("--tune", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    args = p.parse_args()

    cfg = oa.HierarchicalFitConfig(
        draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = oa.filter_main_fit(oa.load_coded(args.input, args.drugs))
    wc = load_word_counts(Path(args.articles))
    df["word_count"] = [wc.get((d, l)) for d, l in zip(df["drug"], df["lang"])]

    # --- 1. length-adjusted hierarchical (existing articles only) -------------
    sub = df[df["word_count"].notna()].copy()
    # log() is undefined at 0. No zero-length article exists in the current
    # snapshot, but report any rather than dropping it silently — a silent drop
    # would misattribute the cell to "no article" in the §5c decomposition.
    zero_len = sub[sub["word_count"] <= 0]
    if not zero_len.empty:
        pairs = sorted(zip(zero_len["drug"], zero_len["lang"]))
        print(f"WARNING: {len(zero_len)} existing article(s) with word_count <= 0 "
              f"excluded from the length model (log undefined): {pairs}")
        sub = sub[sub["word_count"] > 0].copy()
    sub["log_wc"] = np.log(sub["word_count"].astype(float))
    sub["log_wc_c"] = sub["log_wc"] - sub["log_wc"].mean()

    f_base = f"code ~ 1 + {LANG_TERM} + (1 | drug) + ({LANG_TERM} | topic)"
    f_len = f"code ~ 1 + {LANG_TERM} + log_wc_c + (1 | drug) + ({LANG_TERM} | topic)"

    print(f"main-fit rows: {len(df)}; existing articles (with word count): {len(sub)}")
    print("Fitting baseline (same rows, no length covariate) ...")
    id_base = _fit(f_base, sub, "cumulative", cfg)
    print("Fitting length-adjusted ...")
    id_len = _fit(f_len, sub, "cumulative", cfg)

    # Reuse the §5b extractor so the pooled ORs are computed identically.
    base = oa.hierarchical_pooled_ors(id_base).assign(model="baseline")
    leng = oa.hierarchical_pooled_ors(id_len).assign(model="length_adjusted")
    pooled = pd.concat([base, leng], ignore_index=True)[
        ["model", "language", "or_median", "hdi_low", "hdi_high", "prob_or_lt_1"]
    ]
    pooled.to_csv(out / "pooled_ors_by_model.csv", index=False)

    if "log_wc_c" not in id_len.posterior:
        raise KeyError(
            "the length-adjusted posterior carries no `log_wc_c` term (found: "
            f"{sorted(id_len.posterior.data_vars)}). bambi may have renamed or "
            "dropped the covariate; the length adjustment did not take effect."
        )
    beta = id_len.posterior["log_wc_c"].stack(s=("chain", "draw")).values
    or_wc = np.exp(beta)
    pd.DataFrame(
        [
            {
                "term": "log_word_count",
                **oa.or_summary(or_wc, 0.95),
                "prob_beta_gt_0": float((beta > 0).mean()),
            }
        ]
    ).to_csv(out / "log_wc_effect.csv", index=False)

    # --- 2. conditional: P(code 2 | code >= 1) --------------------------------
    cond = df[df["code"] >= 1].copy()
    cond["directive"] = (cond["code"] == 2).astype(int)
    counts = (
        cond.groupby("lang")
        .agg(n_mentioned=("directive", "size"), n_directive=("directive", "sum"))
        .reindex(["en", *LANG_ORDER])
        .reset_index()
        .rename(columns={"lang": "language"})
    )
    # An edition with no mentioned cells would divide 0/0; surface it instead of
    # writing a bare NaN to the CSV.
    empty = counts.loc[counts["n_mentioned"].fillna(0) == 0, "language"].tolist()
    if empty:
        print(f"WARNING: no mentioned (code >= 1) cells for {empty}; "
              "P(directive | mention) is undefined for those editions")
    counts["p_directive_given_mention"] = (
        counts["n_directive"] / counts["n_mentioned"]
    ).round(4)
    counts.to_csv(out / "conditional_counts.csv", index=False)

    print(f"Fitting conditional model on {len(cond)} mentioned cells ...")
    id_cond = _fit(f"directive ~ 1 + {LANG_TERM} + (1 | drug)", cond, "bernoulli", cfg)
    _lang_or_table(id_cond).to_csv(out / "conditional_ors.csv", index=False)

    manifest = {
        "input": str(args.input),
        "n_main_fit_rows": int(len(df)),
        "n_existing_articles": int(len(sub)),
        "n_mentioned_cells": int(len(cond)),
        "seed": args.seed,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "formula_baseline": f_base,
        "formula_length_adjusted": f_len,
        "formula_conditional": f"directive ~ 1 + {LANG_TERM} + (1 | drug)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== Pooled OR vs EN — baseline vs length-adjusted ===")
    print(pooled.round(3).to_string(index=False))
    print("\n=== log(word count) effect ===")
    print(pd.read_csv(out / "log_wc_effect.csv").round(3).to_string(index=False))
    print("\n=== P(directive | risk mentioned) ===")
    print(counts.to_string(index=False))
    print("\n=== Conditional OR vs EN, given a mention ===")
    print(pd.read_csv(out / "conditional_ors.csv").round(3).to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
