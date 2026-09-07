"""Refit the hierarchical model (same config as run_hierarchical_analysis.py,
seed 23) and write 80% credible intervals for the two-level forest plot.

Sanity-checks the refit's 95% summaries against the committed
data/coded/hierarchical_four_topic/{pooled,per_topic}_ors.csv before writing:
    data/coded/hierarchical_four_topic/pooled_ors_80.csv
    data/coded/hierarchical_four_topic/per_topic_ors_80.csv
    paper/data/hier_pooled_ors_80.csv
    paper/data/hier_per_topic_ors_80.csv
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from g6pd.config import DRUGS_CSV, CODED_DIR
from g6pd.ordinal_analysis import (
    HierarchicalFitConfig,
    fit_hierarchical_ordinal,
    hierarchical_per_topic_ors,
    hierarchical_pooled_ors,
    load_coded,
)


def main():
    warnings.filterwarnings("ignore")
    out = CODED_DIR / "hierarchical_four_topic"
    df = load_coded(CODED_DIR / "coded_four_topic.csv", DRUGS_CSV)
    cfg = HierarchicalFitConfig(draws=1500, tune=1500, chains=4, seed=23)
    print(f"Fitting hierarchical model ({cfg.chains} chains x {cfg.draws} draws) ...")
    _, idata = fit_hierarchical_ordinal(df, cfg)

    # sanity check: refit at 95% must reproduce the committed summaries
    pooled95 = hierarchical_pooled_ors(idata, hdi_prob=0.95)
    committed = pd.read_csv(out / "pooled_ors.csv")
    merged = pooled95.merge(committed, on="language", suffixes=("_new", "_old"))
    max_diff = (merged["or_median_new"] - merged["or_median_old"]).abs().max()
    print(f"max |pooled median difference| vs committed: {max_diff:.4f}")
    if max_diff > 0.01:
        sys.exit("Refit does not reproduce the committed medians; aborting.")

    pooled80 = hierarchical_pooled_ors(idata, hdi_prob=0.80)
    per80 = hierarchical_per_topic_ors(idata, hdi_prob=0.80)
    for frame, name in [(pooled80, "pooled_ors_80.csv"), (per80, "per_topic_ors_80.csv")]:
        frame.to_csv(out / name, index=False)
    pooled80.to_csv(ROOT / "paper" / "data" / "hier_pooled_ors_80.csv", index=False)
    per80.to_csv(ROOT / "paper" / "data" / "hier_per_topic_ors_80.csv", index=False)
    print(pooled80.to_string(index=False))
    print(per80.to_string(index=False))
    print("Wrote 80% interval CSVs.")


if __name__ == "__main__":
    main()
