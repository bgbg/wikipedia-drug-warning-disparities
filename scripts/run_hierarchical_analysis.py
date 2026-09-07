"""
run_hierarchical_analysis.py — hierarchical ordinal analysis (M3, #31).

Fits the partial-pooling cumulative-logit model
    code ~ 1 + C(lang, Treatment('en')) + (1 | drug) + (C(lang, Treatment('en')) | topic)
on the combined coded dataset (three topics since PGx-SCAR was dropped 2026-07-01;
the output dir keeps its historical `_four_topic` name) and writes:
    pooled_ors.csv        - pooled non-English-vs-EN OR per language (the headline)
    per_topic_ors.csv     - per (language, topic) OR vs EN (secondary / forest plot)
    tau.csv               - SD of the per-topic language effect ("how systematic")
    convergence.csv       - max R-hat, min ESS, divergences
    manifest.json         - run metadata

Usage:
    python scripts/run_hierarchical_analysis.py \
        --input data/coded/coded_four_topic.csv \
        --output-dir data/coded/hierarchical_four_topic
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arviz as az
import pandas as pd

from g6pd.config import DRUGS_CSV, CODED_DIR
from g6pd.ordinal_analysis import (
    HierarchicalFitConfig,
    fit_hierarchical_ordinal,
    hierarchical_per_topic_ors,
    hierarchical_pooled_ors,
    hierarchical_tau,
    load_coded,
)


def parse_args():
    p = argparse.ArgumentParser(description="Four-topic hierarchical ordinal analysis")
    p.add_argument("--input", default=str(CODED_DIR / "coded_four_topic.csv"))
    p.add_argument("--drugs", default=str(DRUGS_CSV))
    p.add_argument("--output-dir", default=str(CODED_DIR / "hierarchical_four_topic"))
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--draws", type=int, default=1500)
    p.add_argument("--tune", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_coded(args.input, args.drugs)
    cfg = HierarchicalFitConfig(draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed)
    print(f"Fitting hierarchical model ({cfg.chains} chains x {cfg.draws} draws) ...")
    model, idata = fit_hierarchical_ordinal(df, cfg)

    pooled = hierarchical_pooled_ors(idata)
    per_topic = hierarchical_per_topic_ors(idata)
    tau = hierarchical_tau(idata)
    pooled.to_csv(out / "pooled_ors.csv", index=False)
    per_topic.to_csv(out / "per_topic_ors.csv", index=False)
    tau.to_csv(out / "tau.csv", index=False)

    rhat = float(az.rhat(idata).to_array().max())
    ess = float(az.ess(idata).to_array().min())
    div = int(idata.sample_stats.diverging.sum())
    conv = pd.DataFrame([{"max_rhat": rhat, "min_ess_bulk": ess, "divergences": div}])
    conv.to_csv(out / "convergence.csv", index=False)

    (out / "manifest.json").write_text(json.dumps({
        "input": args.input,
        "formula": "code ~ 1 + C(lang, Treatment('en')) + (1|drug) + (C(lang, Treatment('en'))|topic)",
        "seed": args.seed, "draws": args.draws, "tune": args.tune, "chains": args.chains,
        "n_obs": int(len(load_coded(args.input, args.drugs))),
    }, indent=2))

    print("\n=== POOLED OR vs EN (across three topics) ===")
    print(pooled.to_string(index=False))
    print("\n=== tau (per-topic SD of the language effect) ===")
    print(tau.to_string(index=False))
    print("\n=== per-topic OR vs EN ===")
    print(per_topic.to_string(index=False))
    print(f"\nConvergence: max R-hat={rhat:.4f}, min ESS={ess:.0f}, divergences={div}")
    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
