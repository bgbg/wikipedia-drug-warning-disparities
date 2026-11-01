"""
validation_precheck.py: free precheck of instrument agreement from existing data (#77).

Two tables, per language and domain, over the risk-positive cells whose article exists
(the same 85-cell pool the collaborators validate):

1. Inter-model agreement of the three LLM classifiers (unanimity and mean pairwise
   quadratic-weighted kappa), from the per-model codes stored with each coded cell.
   The third model is grok-3 for G6PD and teratogenicity and grok-4.3 for QT; pairs are
   taken by position, so pooled rows treat them as one rater. Pairs whose kappa is
   undefined (no variation in both raters) are left out of the mean.
2. LLM (final code) versus the legacy keyword coder (src/g6pd/coding.py). The keyword
   coder exists for G6PD only, so this table covers the 8 G6PD drugs.

Outputs (committed):
    data/validation/precheck_inter_model_agreement.csv
    data/validation/precheck_llm_vs_keyword.csv

Usage:
    python scripts/validation_precheck.py
"""

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.agreement import weighted_kappa  # noqa: E402
from g6pd.coding import code_article  # noqa: E402
from g6pd.config import (  # noqa: E402
    ARTICLES_JSONL,
    DRUGS_CSV,
    QT_WARNINGS_CSV,
    ROOT_DIR,
    TOPIC_WARNINGS,
    VALIDATION_DIR,
)

LANGS = ["en", "he", "ar", "ru"]
PAPER_DATA = ROOT_DIR / "paper" / "data"
# Adjudicated G6PD / teratogenicity finals live under paper/data (as in
# assemble_four_topic_coded.py); QT is the June-2026 snapshot.
SOURCES = {
    "g6pd": PAPER_DATA / "g6pd_warnings.csv",
    "teratogenicity": PAPER_DATA / "teratogenicity_warnings.csv",
    "qt": QT_WARNINGS_CSV,
}


def positive_cells(domain: str, drugs: pd.DataFrame) -> pd.DataFrame:
    """Risk-positive cells of a domain with an existing article, with per-model codes."""
    risk_col = TOPIC_WARNINGS[domain][1]
    positives = set(drugs.loc[drugs[risk_col].astype(str).str.strip() != "", "drug"])
    df = pd.read_csv(SOURCES[domain])
    df = df[df["drug"].isin(positives) & df["exists"].astype(bool)].copy()
    df["domain"] = domain
    df["codes"] = df["model_codes"].map(lambda s: list(json.loads(s).values()))
    return df[["drug", "lang", "domain", "warning_code", "codes"]]


def inter_model_row(cells: pd.DataFrame) -> dict:
    cols = list(zip(*cells["codes"]))  # one tuple of codes per model
    kappas = [weighted_kappa(cols[i], cols[j]) for i, j in itertools.combinations(range(len(cols)), 2)]
    kappas = [k for k in kappas if k == k]
    unanimous = int(cells["codes"].map(lambda c: len(set(c)) == 1).sum())
    return {
        "n_cells": len(cells),
        "n_unanimous": unanimous,
        "pct_unanimous": round(100 * unanimous / len(cells), 1),
        "mean_pairwise_weighted_kappa": round(sum(kappas) / len(kappas), 3) if kappas else float("nan"),
    }


def inter_model_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in LANGS + ["all"]:
        for domain in ["g6pd", "teratogenicity", "qt", "all"]:
            sub = cells
            if lang != "all":
                sub = sub[sub["lang"] == lang]
            if domain != "all":
                sub = sub[sub["domain"] == domain]
            if len(sub):
                rows.append({"language": lang, "domain": domain, **inter_model_row(sub)})
    return pd.DataFrame(rows)


def keyword_table(g6pd_cells: pd.DataFrame, articles_path: Path) -> pd.DataFrame:
    arts = {}
    with open(articles_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                arts[(r["drug"], r["lang"])] = r
    kw = [code_article(arts[(r.drug, r.lang)])["warning_code"] for r in g6pd_cells.itertuples()]
    df = g6pd_cells.assign(keyword_code=kw)
    rows = []
    for lang in LANGS + ["all"]:
        sub = df if lang == "all" else df[df["lang"] == lang]
        llm, k = sub["warning_code"], sub["keyword_code"]
        rows.append({
            "language": lang,
            "domain": "g6pd",
            "n_cells": len(sub),
            "n_exact_agree": int((llm == k).sum()),
            "pct_exact_agree": round(100 * (llm == k).mean(), 1),
            "weighted_kappa": round(weighted_kappa(llm, k), 3),
            "n_keyword_lower": int((k < llm).sum()),
            "n_keyword_higher": int((k > llm).sum()),
            "n_llm_mentions_keyword_zero": int(((llm >= 1) & (k == 0)).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    drugs = pd.read_csv(DRUGS_CSV).fillna("")
    cells = pd.concat([positive_cells(d, drugs) for d in SOURCES], ignore_index=True)
    inter = inter_model_table(cells)
    kw = keyword_table(cells[cells["domain"] == "g6pd"], ARTICLES_JSONL)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    inter.to_csv(VALIDATION_DIR / "precheck_inter_model_agreement.csv", index=False)
    kw.to_csv(VALIDATION_DIR / "precheck_llm_vs_keyword.csv", index=False)
    print(f"Risk-positive cells with an article: {len(cells)}\n")
    print("Inter-model agreement (3 LLMs)\n" + inter.to_string(index=False))
    print("\nLLM final code vs legacy keyword coder (G6PD only)\n" + kw.to_string(index=False))


if __name__ == "__main__":
    main()
