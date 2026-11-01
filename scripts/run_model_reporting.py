"""
run_model_reporting.py: reporting additions to the primary hierarchical model (#79).

The prespecified primary model is unchanged (formula, priors, seed, sampler settings of
`scripts/run_hierarchical_analysis.py`). This script refits it, checks that the refit reproduces
the committed pooled ORs, and derives from the same posterior draws:

    finite_population_summary.csv       population-level pooled OR beside the finite-population OR
    tau_prior_table.csv                 pooled ORs and tau under three priors on the SD of the
                                        language effect (sensitivity analysis)
    joint_posterior_probabilities.csv   P(all three pooled ORs < 1), P(all nine cells < 1), ... per fit
    post_hoc_common_effect.csv          POST HOC common-effect model with domain fixed effects
    reproduction_check.csv              committed versus refitted pooled ORs
    convergence.csv, manifest.json

Every output is written to --output-dir (default data/coded/hierarchical_reporting).
The 12 QT inter-model disagreements carry the majority-vote code, final for the submitted analysis.

Usage:
    python scripts/run_model_reporting.py

Toolchain note (macOS 15 or later with Xcode Command Line Tools 26): pytensor 2.38 adds `-ld64` to the
clang call, which that linker rejects ("library 'd64' not found"), so any pymc fit fails. The model, priors
and seeds are not involved. Making pytensor see an older macOS version for its flag check compiles normally
and reproduces the committed primary-model medians exactly (checked 3/10/2026):

    python -c "import platform, runpy, sys; platform.mac_ver = lambda *a, **k: ('14.0', ('', '', ''), 'arm64'); \
sys.path.insert(0, 'src'); sys.argv = ['scripts/run_model_reporting.py']; \
runpy.run_path('scripts/run_model_reporting.py', run_name='__main__')"
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arviz as az
import pandas as pd

from g6pd.config import CODED_DIR, DRUGS_CSV
from g6pd.model_reporting import (
    COMMON_EFFECT_FORMULA,
    common_effect_ors,
    finite_population_ors,
    fit_common_effect,
    fit_hierarchical_tau_prior,
    joint_prob_below_one,
    joint_probabilities,
)
from g6pd.ordinal_analysis import (
    HIER_FORMULA,
    HierarchicalFitConfig,
    filter_main_fit,
    fit_hierarchical_ordinal,
    hierarchical_pooled_ors,
    hierarchical_tau,
    load_coded,
)

COMMITTED_POOLED = CODED_DIR / "hierarchical_four_topic" / "pooled_ors.csv"
ALT_TAU_SDS = (0.5, 1.0)  # HalfNormal scales tried besides the bambi default of the primary fit
REPRO_TOL = 1e-6  # floating-point serialisation only: the refit must reproduce the committed medians


def parse_args():
    p = argparse.ArgumentParser(description="Reporting additions to the hierarchical primary model")
    p.add_argument("--input", default=str(CODED_DIR / "coded_four_topic.csv"))
    p.add_argument("--drugs", default=str(DRUGS_CSV))
    p.add_argument("--output-dir", default=str(CODED_DIR / "hierarchical_reporting"))
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--draws", type=int, default=1500)
    p.add_argument("--tune", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--repro-tol", type=float, default=REPRO_TOL,
                   help="max absolute difference between refitted and committed pooled medians")
    return p.parse_args()


def convergence_row(fit: str, idata) -> dict:
    return {
        "fit": fit,
        "max_rhat": float(az.rhat(idata).to_array().max()),
        "min_ess_bulk": float(az.ess(idata).to_array().min()),
        "divergences": int(idata.sample_stats.diverging.sum()),
    }


def default_tau_sd(model) -> float:
    """HalfNormal scale bambi assigns by default to the SD of the language effect (the primary tau prior)."""
    m = re.search(r"\)\|topic ~ Normal\(mu: 0\.0, sigma: HalfNormal\(sigma: \[?([\d.]+)", str(model))
    if not m:
        raise RuntimeError("could not read the default tau prior from the fitted model")
    return float(m.group(1))


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_coded(args.input, args.drugs)
    cfg = HierarchicalFitConfig(draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed)
    convergence = []

    # 1. Primary fit and reproduction check (gate: nothing else is trusted if this fails).
    print("Fitting the primary hierarchical model ...")
    model, primary = fit_hierarchical_ordinal(df, cfg)
    primary_tau_sd = default_tau_sd(model)
    convergence.append(convergence_row("primary", primary))
    pooled = hierarchical_pooled_ors(primary)
    committed = pd.read_csv(COMMITTED_POOLED).set_index("language")
    repro = pooled.set_index("language")[["or_median"]].rename(columns={"or_median": "refit_or_median"})
    repro["committed_or_median"] = committed["or_median"]
    repro["abs_diff"] = (repro["refit_or_median"] - repro["committed_or_median"]).abs()
    repro["reproduced"] = repro["abs_diff"] <= args.repro_tol
    repro.reset_index().to_csv(out / "reproduction_check.csv", index=False)
    print(repro.to_string())
    if not repro["reproduced"].all():
        sys.exit("Reproduction check failed: the refit does not match the committed pooled ORs.")

    # 2. Finite-population summary.
    finite = finite_population_ors(primary)
    finite.to_csv(out / "finite_population_summary.csv", index=False)

    # 3. Tau prior sensitivity (primary = bambi default) and 4. joint probabilities under each prior.
    fits = {f"HalfNormal({primary_tau_sd:g}) bambi default, primary": (primary_tau_sd, primary)}
    for sd in ALT_TAU_SDS:
        print(f"Refitting with HalfNormal({sd:g}) on tau ...")
        _, idata = fit_hierarchical_tau_prior(df, sd, cfg)
        fits[f"HalfNormal({sd:g})"] = (sd, idata)
        convergence.append(convergence_row(f"tau_prior_halfnormal_{sd:g}", idata))
    tau_rows, joint_rows = [], []
    for label, (sd, idata) in fits.items():
        pooled_p = hierarchical_pooled_ors(idata).set_index("language")
        tau_p = hierarchical_tau(idata).set_index("language")
        for lang in pooled_p.index:
            tau_rows.append({
                "prior": label, "tau_prior_sd": sd, "language": lang, **pooled_p.loc[lang].to_dict(),
                "tau_median": tau_p.loc[lang, "tau_median"],
                "tau_hdi_low": tau_p.loc[lang, "hdi_low"], "tau_hdi_high": tau_p.loc[lang, "hdi_high"],
            })
        for _, r in joint_probabilities(idata).iterrows():
            joint_rows.append({"fit": label, **r.to_dict()})
    pd.DataFrame(tau_rows).to_csv(out / "tau_prior_table.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(out / "joint_posterior_probabilities.csv", index=False)

    # 5. Post hoc common-effect model.
    print("Fitting the POST HOC common-effect model ...")
    _, common = fit_common_effect(df, cfg)
    convergence.append(convergence_row("post_hoc_common_effect", common))
    ors, draws = common_effect_ors(common)
    ors.insert(0, "model", COMMON_EFFECT_FORMULA)
    ors.insert(0, "post_hoc", True)
    ors["joint_prob_all_three_lt_1"] = joint_prob_below_one(draws)
    ors.to_csv(out / "post_hoc_common_effect.csv", index=False)

    pd.DataFrame(convergence).to_csv(out / "convergence.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "input": args.input,
        "primary_formula": HIER_FORMULA,
        "post_hoc_formula": COMMON_EFFECT_FORMULA,
        "post_hoc_note": "post hoc, not prespecified; no preregistration",
        "primary_tau_prior": f"HalfNormal({primary_tau_sd:g}) (bambi default)",
        "alt_tau_priors": [f"HalfNormal({s:g})" for s in ALT_TAU_SDS],
        "seed": args.seed, "draws": args.draws, "tune": args.tune, "chains": args.chains,
        "target_accept": cfg.target_accept, "prior_sd_language": cfg.prior_sd,
        "n_input_rows": int(len(df)),
        "n_obs_fitted": int(len(filter_main_fit(df))),
        "repro_tol": args.repro_tol,
        "qt_status": "QT disagreements carry the majority-vote code, final for the submitted analysis (decision of 15/10/2026)",
    }, indent=2))

    print("\n=== finite-population summary ===\n", finite.to_string(index=False))
    print("\n=== tau prior table ===\n", pd.DataFrame(tau_rows).round(3).to_string(index=False))
    print("\n=== joint posterior probabilities ===\n", pd.DataFrame(joint_rows).round(3).to_string(index=False))
    print("\n=== post hoc common-effect model ===\n", ors.round(3).to_string(index=False))
    print("\n=== convergence ===\n", pd.DataFrame(convergence).round(4).to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
