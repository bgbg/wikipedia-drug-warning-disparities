"""Tests for scripts/validation_precheck.py (#77) and src/g6pd/agreement.py."""
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import validation_precheck as vp  # noqa: E402
from g6pd.agreement import weighted_kappa  # noqa: E402


def test_weighted_kappa_known_values():
    assert weighted_kappa([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]) == 1.0
    assert weighted_kappa([0, 0, 2, 2], [2, 2, 0, 0]) == -1.0
    assert math.isnan(weighted_kappa([0, 0, 0], [0, 0, 0]))  # no variation: undefined
    assert math.isnan(weighted_kappa([], []))
    # a one-step disagreement costs less than a two-step one
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 1]) > weighted_kappa([0, 1, 2, 2], [0, 1, 2, 0])


def test_inter_model_row_unanimity_and_kappa():
    cells = pd.DataFrame({"codes": [[0, 0, 0], [2, 2, 2], [1, 1, 2], [0, 1, 1]]})
    row = vp.inter_model_row(cells)
    assert row["n_cells"] == 4 and row["n_unanimous"] == 2 and row["pct_unanimous"] == 50.0
    assert -1 <= row["mean_pairwise_weighted_kappa"] <= 1


def test_inter_model_table_has_language_and_overall_rows():
    cells = pd.DataFrame({
        "drug": list("abcd"), "lang": ["en", "he", "ar", "ru"], "domain": ["g6pd", "qt", "qt", "teratogenicity"],
        "warning_code": [2, 0, 1, 2], "codes": [[2, 2, 2], [0, 0, 0], [1, 1, 0], [2, 2, 2]],
    })
    t = vp.inter_model_table(cells)
    assert {"en", "he", "ar", "ru", "all"} <= set(t["language"])
    overall = t[(t["language"] == "all") & (t["domain"] == "all")].iloc[0]
    assert overall["n_cells"] == 4 and overall["n_unanimous"] == 3
