"""
adjudication_sensitivity.py — bracket the provisional result (M3).

The QT domain has inter-model disagreements awaiting pharmacist adjudication;
meanwhile they carry a provisional majority-vote code. (PGx-SCAR was dropped
2026-07-01 — see methodological_decisions.md — so it is no longer bracketed.)
This script bounds how much the headline could move once Maya adjudicates, by
refitting the hierarchical model under three scenarios for the disagreement cells:

    min      - every disagreement set to the LOWEST of its valid model codes
    majority - the provisional majority-vote codes actually in coded_four_topic.csv
    max      - every disagreement set to the HIGHEST of its valid model codes

Because Maya's adjudicated code for each cell must lie within the models' range,
the min/max fits bound the true result. If the pooled non-English-vs-EN odds
ratios stay below 1 under both extremes, the conclusion is robust to whatever
the adjudication decides.

Usage:
    conda run -n pharmaco python scripts/adjudication_sensitivity.py
"""

import csv
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from g6pd.config import CODED_DIR, DRUGS_CSV, LLM_DISAGREEMENTS_CSV
from g6pd.ordinal_analysis import (
    HierarchicalFitConfig,
    fit_hierarchical_ordinal,
    hierarchical_pooled_ors,
    load_coded,
)

COMBINED = CODED_DIR / "coded_four_topic.csv"
# PGx-SCAR dropped 2026-07-01; QT is the only new topic still adjudicated.
NEW_TOPICS = {"qt"}


def disagreement_bounds() -> dict:
    """{(drug, lang, topic): (min_code, max_code)} from valid model codes."""
    bounds = {}
    with open(LLM_DISAGREEMENTS_CSV) as fh:
        for r in csv.DictReader(fh):
            if r.get("topic") not in NEW_TOPICS:
                continue
            codes = [int(r[f"code_{i}"]) for i in (1, 2, 3)
                     if r.get(f"code_{i}") not in (None, "", "nan") and int(r[f"code_{i}"]) >= 0]
            if codes:
                bounds[(r["drug"], r["lang"], r["topic"])] = (min(codes), max(codes))
    return bounds


def fit_scenario(base: pd.DataFrame, bounds: dict, which: str) -> pd.DataFrame:
    df = base.copy()
    if which in ("min", "max"):
        idx = 0 if which == "min" else 1
        key = df[["drug", "lang", "topic"]].apply(tuple, axis=1)
        for i, k in key.items():
            if k in bounds:
                df.at[i, "final_code"] = bounds[k][idx]
    tmp = CODED_DIR / f"_sens_{which}.csv"
    df.to_csv(tmp, index=False)
    coded = load_coded(tmp, DRUGS_CSV)
    tmp.unlink()
    _, idata = fit_hierarchical_ordinal(coded, HierarchicalFitConfig(draws=1200, tune=1200, chains=4))
    out = hierarchical_pooled_ors(idata)
    out.insert(0, "scenario", which)
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    base = pd.read_csv(COMBINED)
    bounds = disagreement_bounds()
    print(f"{len(bounds)} new-domain disagreement cells bracketed.\n")

    results = [fit_scenario(base, bounds, w) for w in ("min", "majority", "max")]
    table = pd.concat(results, ignore_index=True)
    out_csv = CODED_DIR / "hierarchical_four_topic" / "adjudication_sensitivity.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    print("=== Pooled OR vs EN under each adjudication scenario ===")
    show = table[["scenario", "language", "or_median", "hdi_low", "hdi_high", "prob_or_lt_1"]]
    print(show.to_string(index=False))
    # Verdict: does the gap (OR<1) survive both extremes?
    worst = table.groupby("language")["or_median"].max()
    print("\nWorst-case (largest) pooled OR per language across scenarios:")
    for lang, v in worst.items():
        print(f"  {lang}: {v:.3f}  ->  {'gap holds (OR<1)' if v < 1 else 'gap NOT robust'}")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
