"""
model_agreement_table.py: inter-model agreement of the three LLM classifiers, by language and domain (#79).

Covers every cell whose article exists (risk-positive drugs and negative controls), for the three
analysed domains. It uses only the per-model codes stored with each classification, so it does not
depend on any pharmacist adjudication. Cells with no article are excluded: the models return 0 on an
empty article, which would inflate agreement. Columns mirror
`data/validation/precheck_inter_model_agreement.csv`, which covers the 85 risk-positive cells only.

Output (committed): data/coded/model_agreement_three_domains.csv

Usage:
    python scripts/model_agreement_table.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.config import CODED_DIR  # noqa: E402
from validation_precheck import LANGS, SOURCES, inter_model_row  # noqa: E402

DOMAINS = ["g6pd", "teratogenicity", "qt"]
OUT = CODED_DIR / "model_agreement_three_domains.csv"


def classified_cells() -> pd.DataFrame:
    """Cells with an existing article, with the list of the three models' codes."""
    parts = []
    for domain in DOMAINS:
        df = pd.read_csv(SOURCES[domain])
        df = df[df["exists"].astype(bool)].copy()
        df["domain"] = domain
        df["codes"] = df["model_codes"].map(lambda s: list(json.loads(s).values()))
        parts.append(df[["drug", "lang", "domain", "codes"]])
    return pd.concat(parts, ignore_index=True)


def agreement_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in LANGS + ["all"]:
        for domain in DOMAINS + ["all"]:
            sub = cells[(cells["lang"] == lang) | (lang == "all")]
            sub = sub[(sub["domain"] == domain) | (domain == "all")]
            rows.append({"language": lang, "domain": domain, **inter_model_row(sub)})
    return pd.DataFrame(rows)


def main():
    cells = classified_cells()
    table = agreement_table(cells)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {OUT} ({len(cells)} cells with an article)")


if __name__ == "__main__":
    main()
