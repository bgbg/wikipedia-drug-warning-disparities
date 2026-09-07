"""Run the ordinal-aware analysis pipeline (issue #23).

Reads the coded LLM-classification CSV plus the drug catalogue, fits the
Bayesian cumulative-logit model with a drug random effect, runs three
robustness checks (frequentist OrderedModel + cluster-robust SEs, Bayesian
prior sensitivity, paired Wilcoxon), computes Cliff's delta, validates the
negative controls, and writes all outputs (CSVs, figures, posterior trace,
validation notes) under `--output-dir`.

This script produces the per-cell (unpooled) ordinal analysis and robustness
battery consumed by the `paper/` manuscript. It is run on the combined coded
CSV; see `scripts/run_hierarchical_analysis.py` for the primary three-domain
partial-pooling model.

Example:
    python scripts/run_ordinal_analysis.py \\
        --input data/coded/llm_classifications.csv \\
        --output-dir data/coded/ordinal_analysis_15drug \\
        --seed 23
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g6pd.config import DRUGS_CSV, LLM_CLASSIFICATIONS_CSV  # noqa: E402
from g6pd import ordinal_analysis as oa  # noqa: E402
from g6pd import ordinal_plots as op  # noqa: E402


def _repo_relative(path: Path) -> str:
    """Return `path` relative to the repo root when possible, else just the name.

    Avoids leaking absolute or worktree-specific paths into committed
    artefacts (manifest.json, validation_notes.md).
    """
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return p.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ordinal-aware analysis pipeline (#23)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=LLM_CLASSIFICATIONS_CSV,
        help="Path to coded LLM classifications CSV.",
    )
    p.add_argument(
        "--drugs",
        type=Path,
        default=DRUGS_CSV,
        help="Path to drugs.csv (drug catalogue with category column).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write CSVs, figures, posterior trace, and notes.",
    )
    p.add_argument(
        "--topics",
        nargs="+",
        choices=list(oa.TOPICS),
        default=None,
        help=(
            "Restrict the per-cell model to a subset of topics (default: all "
            "three). Must include `g6pd` — the per-cell baseline machinery uses "
            "`ar:g6pd` as the implicit reference cell. Pass e.g. "
            "`--topics g6pd teratogenicity` to reproduce the two-domain "
            "Results-§5 analysis as a frozen artifact."
        ),
    )
    p.add_argument("--seed", type=int, default=23, help="PRNG seed.")
    p.add_argument(
        "--draws", type=int, default=2000, help="Posterior draws per chain."
    )
    p.add_argument("--tune", type=int, default=2000, help="Tuning draws per chain.")
    p.add_argument("--chains", type=int, default=4, help="Number of chains.")
    p.add_argument(
        "--target-accept",
        type=float,
        default=0.95,
        help="NUTS target acceptance.",
    )
    p.add_argument(
        "--prior-sd",
        type=float,
        default=1.5,
        help="Default sigma for cell-effect prior.",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Cliff's delta cluster-bootstrap iterations.",
    )
    p.add_argument(
        "--prior-sensitivity",
        nargs="+",
        type=float,
        default=[0.5, 1.5, 5.0],
        help="prior_sd values for prior sensitivity refits.",
    )
    p.add_argument(
        "--rope-low", type=float, default=0.5, help="ROPE lower bound on OR scale."
    )
    p.add_argument(
        "--rope-high", type=float, default=2.0, help="ROPE upper bound on OR scale."
    )
    return p.parse_args()


def main() -> int:
    # Narrow filters: silence known-noisy categories from third-party libs but
    # keep PyMC sampling diagnostics (divergences, max-tree-depth, R-hat warnings)
    # visible — those signal real fit problems.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="formulae.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas.*")
    args = parse_args()

    # Optional topic subset. The per-cell helpers iterate the module-level
    # `oa.TOPICS`/`oa.TOPIC_RISK_COL` and `.sel(topic=...)` on the fitted xarray,
    # so a subset run requires restricting both constants before the pipeline
    # runs (and filtering the loaded frame to match). This is the supported
    # "two-domain mode" used to freeze the Results-§5 artifact.
    selected_topics = tuple(args.topics) if args.topics else oa.TOPICS
    if "g6pd" not in selected_topics:
        # The per-cell baseline machinery hard-codes ar:g6pd as the implicit
        # reference cell (oa._lang_topic_effects) and the frequentist model
        # treats g6pd as the reference topic; a g6pd-excluding subset would
        # otherwise crash only after the (expensive) Bayesian fit. Fail fast.
        sys.exit(
            "ERROR: --topics must include 'g6pd' (the per-cell baseline uses "
            f"ar:g6pd as the reference cell); got: {', '.join(selected_topics)}"
        )
    if set(selected_topics) != set(oa.TOPICS):
        oa.TOPICS = selected_topics
        oa.TOPIC_RISK_COL = {t: oa.TOPIC_RISK_COL[t] for t in selected_topics}

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"# Ordinal-aware analysis pipeline (#23)")
    print(f"  input         : {args.input}")
    print(f"  drugs         : {args.drugs}")
    print(f"  output-dir    : {out}")
    print(f"  topics        : {', '.join(selected_topics)}")
    print(f"  seed          : {args.seed}")
    print(f"  draws/chains  : {args.draws} / {args.chains}")
    print()

    # ------------------------------------------------------------------
    # 1. Load + filter
    # ------------------------------------------------------------------
    df = oa.load_coded(args.input, args.drugs)
    df = df[df["topic"].isin(selected_topics)].reset_index(drop=True)
    main = oa.filter_main_fit(df)
    print(
        f"Loaded {len(df)} rows ({len(selected_topics)} topic(s)); "
        f"main-fit subset = {len(main)} "
        f"(study drugs only, drug-category-matched topic)."
    )

    # Persist the prepared dataset so #24 can reuse the same view.
    main.to_csv(out / "main_fit_input.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Negative-control validation
    # ------------------------------------------------------------------
    controls = oa.validate_controls(df)
    controls.to_csv(out / "control_validation.csv", index=False)
    deviates = controls[controls["deviates"]]
    unexplained = deviates[~deviates["documented_exception"]]
    print(
        f"Controls: {len(controls)} article-rows; {len(deviates)} non-zero "
        f"({len(deviates) - len(unexplained)} pre-documented, "
        f"{len(unexplained)} unexplained)."
    )

    # ------------------------------------------------------------------
    # 3. Observed composition + descriptive stacked-bar
    # ------------------------------------------------------------------
    observed = oa.observed_composition(main)
    observed.to_csv(out / "observed_composition.csv", index=False)
    fig_obs = op.stacked_composition_plot(observed)
    fig_obs.savefig(fig_dir / "stacked_composition.png", dpi=150, bbox_inches="tight")
    print("Wrote stacked_composition.png")

    # ------------------------------------------------------------------
    # 4. Prior predictive (analytical, by prior_sd)
    # ------------------------------------------------------------------
    prior_or_samples = {
        f"sd={sd}": oa.prior_implied_or_samples(sd, n=20_000, seed=args.seed)
        for sd in args.prior_sensitivity
    }
    fig_prior = op.prior_predictive_summary_plot(prior_or_samples)
    fig_prior.savefig(
        fig_dir / "prior_predictive_or.png", dpi=150, bbox_inches="tight"
    )
    print("Wrote prior_predictive_or.png")

    # ------------------------------------------------------------------
    # 5. Bayesian primary fit
    # ------------------------------------------------------------------
    cfg = oa.BayesianFitConfig(
        prior_sd=args.prior_sd,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
        include_drug_random_effect=True,
    )
    print("Fitting Bayesian primary model (cumulative-logit + drug RE)...")
    model, idata = oa.fit_bayesian_ordinal(main, cfg)

    idata.to_netcdf(out / "posterior.nc")
    print(f"Saved posterior trace to {out / 'posterior.nc'}")

    diag = oa.convergence_diagnostics(idata)
    diag.to_csv(out / "convergence_diagnostics.csv", index=False)
    rhat_max = float(diag["r_hat"].max())
    ess_min = float(diag["ess_bulk"].min())
    print(f"Convergence: max R-hat = {rhat_max:.3f}, min ESS bulk = {ess_min:.0f}")

    # ------------------------------------------------------------------
    # 6. Posterior summaries
    # ------------------------------------------------------------------
    or_table = oa.cumulative_or_vs_reference(idata)
    or_table.to_csv(out / "cumulative_ors.csv", index=False)

    dom = oa.prob_stochastic_dominance(idata)
    dom.to_csv(out / "prob_dominance.csv", index=False)

    inter = oa.interaction_contrast(idata)
    inter.to_csv(out / "interaction_contrast.csv", index=False)

    rope = oa.rope_summary(idata, rope_or=(args.rope_low, args.rope_high))
    rope.to_csv(out / "rope_summary.csv", index=False)

    ppc = oa.posterior_predictive_composition(idata, main)
    ppc.to_csv(out / "posterior_predictive_composition.csv", index=False)

    fig_ppc = op.posterior_predictive_overlay_plot(observed, ppc)
    fig_ppc.savefig(
        fig_dir / "posterior_predictive_overlay.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Wrote posterior_predictive_overlay.png")

    fig_or = op.or_forest_plot(or_table)
    fig_or.savefig(fig_dir / "or_forest.png", dpi=150, bbox_inches="tight")
    print("Wrote or_forest.png")

    # ------------------------------------------------------------------
    # 7. Robustness check: frequentist OrderedModel + cluster-robust SEs
    # ------------------------------------------------------------------
    print("Fitting frequentist OrderedModel + cluster-robust SEs...")
    freq = oa.fit_frequentist_ordered(main)
    freq["ors"].to_csv(out / "frequentist_ors.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Robustness check: Bayesian prior sensitivity
    # ------------------------------------------------------------------
    print(
        "Refitting Bayesian model at multiple prior_sd values for sensitivity..."
    )
    sens = oa.prior_sensitivity_table(
        main, prior_sds=args.prior_sensitivity, base_config=cfg
    )
    sens.to_csv(out / "prior_sensitivity.csv", index=False)

    # ------------------------------------------------------------------
    # 9. Sensitivity: Bayesian without random effect
    # ------------------------------------------------------------------
    print("Refitting Bayesian model without `(1 | drug)`...")
    _, idata_no_re = oa.fit_no_random_effect(main, base_config=cfg)
    or_no_re = oa.cumulative_or_vs_reference(idata_no_re)
    or_no_re["model"] = "no_random_effect"
    or_main_tagged = or_table.copy()
    or_main_tagged["model"] = "primary"
    sens_no_re = pd.concat([or_main_tagged, or_no_re], ignore_index=True)
    sens_no_re.to_csv(out / "sensitivity_no_random_effect.csv", index=False)

    # ------------------------------------------------------------------
    # 10. Robustness check: paired Wilcoxon
    # ------------------------------------------------------------------
    wilcoxon = oa.paired_wilcoxon_table(main)
    wilcoxon.to_csv(out / "wilcoxon_results.csv", index=False)

    # ------------------------------------------------------------------
    # 11. Effect size: Cliff's delta + cluster bootstrap
    # ------------------------------------------------------------------
    print(f"Computing Cliff's delta with {args.n_boot} cluster-bootstrap reps...")
    cliffs = oa.cliffs_delta_table(main, n_boot=args.n_boot, seed=args.seed)
    cliffs.to_csv(out / "cliffs_delta.csv", index=False)

    # ------------------------------------------------------------------
    # 12. Validation notes
    # ------------------------------------------------------------------
    notes = build_validation_notes(
        rhat_max=rhat_max,
        ess_min=ess_min,
        controls_deviates=deviates,
        observed=observed,
        ppc=ppc,
        or_table=or_table,
        freq_ors=freq["ors"],
        sens=sens,
        wilcoxon=wilcoxon,
        cliffs=cliffs,
        args=args,
    )
    (out / "validation_notes.md").write_text(notes)
    print(f"Wrote validation_notes.md ({len(notes)} chars)")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest = {
        "input": _repo_relative(args.input),
        "drugs": _repo_relative(args.drugs),
        "output_dir": _repo_relative(out),
        "topics": list(selected_topics),
        "n_main_fit_rows": int(len(main)),
        "n_control_rows": int(len(controls)),
        "n_control_deviates": int(len(deviates)),
        "max_rhat": rhat_max,
        "min_ess_bulk": ess_min,
        "config": {
            "prior_sd": args.prior_sd,
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "target_accept": args.target_accept,
            "seed": args.seed,
            "n_boot_cliffs": args.n_boot,
            "prior_sensitivity_sds": list(args.prior_sensitivity),
            "rope_or_bounds": [args.rope_low, args.rope_high],
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print()
    print("Done.")
    return 0


def build_validation_notes(
    *,
    rhat_max: float,
    ess_min: float,
    controls_deviates: pd.DataFrame,
    observed: pd.DataFrame,
    ppc: pd.DataFrame,
    or_table: pd.DataFrame,
    freq_ors: pd.DataFrame,
    sens: pd.DataFrame,
    wilcoxon: pd.DataFrame,
    cliffs: pd.DataFrame,
    args,
) -> str:
    """Compose a markdown validation summary covering the acceptance criteria."""

    lines = []
    lines.append("# Ordinal-aware analysis — validation notes")
    lines.append("")
    lines.append(
        f"Pipeline run on `{_repo_relative(args.input)}` with seed = {args.seed}, "
        f"{args.chains} chains × {args.draws} draws."
    )
    lines.append("")

    lines.append("## Bayesian convergence")
    rhat_pass = rhat_max < 1.01
    ess_pass = ess_min > 400
    lines.append(
        f"- Max R-hat: **{rhat_max:.4f}** "
        f"({'PASS' if rhat_pass else 'FAIL'} — threshold 1.01)"
    )
    lines.append(
        f"- Min bulk ESS: **{ess_min:.0f}** "
        f"({'PASS' if ess_pass else 'FAIL'} — threshold 400)"
    )
    lines.append("")

    lines.append("## Posterior predictive coverage")
    obs_prop = observed.set_index(["language", "topic", "code"])["proportion"]
    ppc_idx = ppc.set_index(["language", "topic", "code"])
    miss = []
    for key, p in obs_prop.items():
        if key not in ppc_idx.index:
            continue
        row = ppc_idx.loc[key]
        if not (row["hdi_low"] - 1e-6 <= p <= row["hdi_high"] + 1e-6):
            miss.append((key, float(p), float(row["hdi_low"]), float(row["hdi_high"])))
    n_cells = len(obs_prop)
    lines.append(f"- {n_cells} cells checked; {len(miss)} outside the 95% HDI.")
    if miss:
        lines.append("- Cells outside 95% HDI:")
        for key, p, low, high in miss:
            lg, tp, code = key
            lines.append(
                f"  - {lg} / {tp} / code={code}: observed {p:.3f}, "
                f"HDI [{low:.3f}, {high:.3f}]"
            )
    lines.append("")

    lines.append("## Frequentist robustness (statsmodels OrderedModel + cluster-robust SE)")
    join = or_table.merge(
        freq_ors,
        on=["language", "topic"],
        suffixes=("_bayes", "_freq"),
    )
    lines.append("| language | topic | OR (Bayes mean) | OR (freq) | sign agree |")
    lines.append("|---|---|---|---|---|")
    for _, r in join.iterrows():
        sign_b = np.sign(np.log(r["mean"])) if r["mean"] > 0 else 0
        sign_f = np.sign(r["log_or"])
        agree = "yes" if sign_b == sign_f else "no"
        lines.append(
            f"| {r['language']} | {r['topic']} | {r['mean']:.3g} | "
            f"{r['or_estimate']:.3g} | {agree} |"
        )
    lines.append("")
    extreme = join[(join["or_estimate"] < 1e-3) | (join["or_estimate"] > 1e3)]
    if not extreme.empty:
        lines.append(
            "**Note:** the frequentist fit produces extreme ORs in cells where the "
            "data hit a coding-scale boundary (e.g., all EN teratogenicity codes = 2). "
            "This is complete-separation in ML estimation; the Bayesian primary "
            "fit regularizes via the prior. The Bayesian estimates are the "
            "headline; the frequentist column is informative about data sparsity, "
            "not a contradicting estimate. Sign agreement is the relevant criterion "
            "for this robustness check at small N."
        )
        lines.append("")

    lines.append("## Prior sensitivity")
    pivot = sens.pivot_table(
        index=["language", "topic"],
        columns="prior_sd",
        values="mean",
    )
    lines.append("Posterior-mean OR by prior_sd:")
    lines.append("")
    lines.append(pivot.round(3).to_markdown())
    lines.append("")
    spread = pivot.max(axis=1) / pivot.min(axis=1)
    max_spread = float(spread.max())
    lines.append(
        f"Worst max/min ratio across the prior_sd grid: **{max_spread:.2f}** "
        f"({'stable' if max_spread < 5.0 else 'sensitive'} — heuristic threshold 5x)."
    )
    lines.append("")

    lines.append("## Paired Wilcoxon (distributional cross-check)")
    lines.append("")
    lines.append(wilcoxon.to_markdown(index=False))
    lines.append("")

    lines.append("## Cliff's delta (clinician-facing effect size)")
    lines.append("")
    lines.append(cliffs.to_markdown(index=False))
    lines.append("")

    lines.append("## Negative-control validation")
    unexplained = controls_deviates[~controls_deviates["documented_exception"]]
    pre_doc = controls_deviates[controls_deviates["documented_exception"]]
    if controls_deviates.empty:
        lines.append("- All control article-rows score 0 as expected.")
    else:
        lines.append(
            f"- {len(controls_deviates)} non-zero control rows: "
            f"{len(pre_doc)} pre-documented (see `methodological_decisions.md`), "
            f"{len(unexplained)} unexplained "
            f"({'PASS' if unexplained.empty else 'INVESTIGATE'})."
        )
        lines.append("")
        lines.append(controls_deviates.to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
