"""Regression tests locking the figure/table inputs to source data (audit #43).

No plotting. These assert that the staged figure inputs under paper/data/*.csv
— which paper/make_figures.py renders and paper/03_results.md tabulates — agree
with what is recomputed from the frozen sources:

  * data/coded/coded_four_topic.csv  (the #38/#39-verified warning codes)
  * data/raw/articles.jsonl          (the #36-verified structural counts)

and that the canonical edition order (en, ru, ar, he) is used everywhere.

Once green, future edits cannot silently desync a figure input from its source
or reorder the editions.
"""
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PDATA = REPO / "paper" / "data"
ARTICLES = REPO / "data" / "raw" / "articles.jsonl"
CODED = REPO / "data" / "coded" / "coded_four_topic.csv"

LANG_ORDER = ["en", "ru", "ar", "he"]
NONEN_ORDER = ["ru", "ar", "he"]
# PGx-SCAR dropped as an analysed topic 2026-07-01 (see methodological_decisions.md).
TOPIC_RISK = {
    "g6pd": ("g6pd_warnings.csv", "cpic_risk"),
    "teratogenicity": ("teratogenicity_warnings.csv", "pregnancy_risk"),
    "qt": ("qt_warnings.csv", "qt_risk"),
}


def _read(fn):
    return list(csv.DictReader(open(PDATA / fn)))


def _positives(topic):
    fn, rc = TOPIC_RISK[topic]
    return [r for r in _read(fn) if str(r.get(rc, "")).strip()]


@pytest.fixture(scope="module")
def cft():
    return list(csv.DictReader(open(CODED)))


# --- 1. Figure-input warning codes == coded_four_topic final_code --------------
def test_warning_codes_match_coded_four_topic(cft):
    lookup = {(r["drug"], r["lang"], r["topic"]): r["final_code"] for r in cft}
    for topic in TOPIC_RISK:
        for r in _positives(topic):
            key = (r["drug"], r["lang"], topic)
            assert key in lookup, f"missing in coded_four_topic: {key}"
            assert lookup[key] == r["warning_code"], f"code mismatch {key}"


# --- 2. Results §3 / §4 code-2-of-8 tables -------------------------------------
def test_code2_tables():
    expected = {
        "g6pd": {"en": 6, "ru": 2, "ar": 2, "he": 4},
        "teratogenicity": {"en": 6, "ru": 4, "ar": 4, "he": 4},
    }
    for topic, exp in expected.items():
        got = Counter()
        drugs = set()
        for r in _positives(topic):
            drugs.add(r["drug"])
            if int(r["warning_code"]) == 2:
                got[r["lang"]] += 1
        assert len(drugs) == 8, f"{topic}: expected 8 positive drugs, got {len(drugs)}"
        assert {l: got[l] for l in LANG_ORDER} == exp


# --- 3. observed_composition.csv == composition of coded_four_topic positives ---
def test_observed_composition(cft):
    posdrugs = {t: {r["drug"] for r in _positives(t)} for t in ("g6pd", "teratogenicity")}
    comp = Counter()
    nobs = Counter()
    for r in cft:
        t = r["topic"]
        if t not in posdrugs or r["drug"] not in posdrugs[t]:
            continue
        comp[(r["lang"], t, int(r["final_code"]))] += 1
        nobs[(r["lang"], t)] += 1
    staged = {(r["language"], r["topic"], int(r["code"])): (int(r["count"]), float(r["proportion"]), int(r["n_obs"]))
              for r in _read("observed_composition.csv")}
    for (lang, t), n in nobs.items():
        for c in (0, 1, 2):
            got = comp.get((lang, t, c), 0)
            scount, sprop, snobs = staged[(lang, t, c)]
            assert scount == got, f"{lang}/{t}/{c}: count {scount} != {got}"
            assert round(sprop, 3) == round(got / n, 3)
            assert snobs == n


# --- 4. structural_quality.csv == medians over the 25-drug set -----------------
def test_structural_quality():
    drugs25 = ({r["drug"] for r in _read("g6pd_warnings.csv")} |
               {r["drug"] for r in _read("teratogenicity_warnings.csv")})
    assert len(drugs25) == 25
    by = defaultdict(lambda: {"wc": [], "nr": [], "cd": []})
    for line in open(ARTICLES):
        r = json.loads(line)
        if r["drug"] not in drugs25 or not r.get("exists"):
            continue
        wc, nr = r.get("word_count"), r.get("n_references")
        if wc is None or nr is None:
            continue
        by[r["lang"]]["wc"].append(wc)
        by[r["lang"]]["nr"].append(nr)
        by[r["lang"]]["cd"].append(nr / wc * 1000 if wc else 0)
    staged = {r["lang"]: r for r in _read("structural_quality.csv")}
    for lang in LANG_ORDER:
        s = staged[lang]
        assert int(float(s["median_word_count"])) == int(statistics.median(by[lang]["wc"]))
        # references median stored as an int floor (AR true median 10.5 -> 10).
        assert int(float(s["median_references"])) == int(statistics.median(by[lang]["nr"]))
        assert round(float(s["median_citation_density"]), 1) == round(statistics.median(by[lang]["cd"]), 1)


def test_hebrew_reference_fix():
    # The corrected Hebrew values: 4 refs / 7.1 density, NOT the pre-fix zero.
    he = {r["lang"]: r for r in _read("structural_quality.csv")}["he"]
    assert int(float(he["median_references"])) == 4
    assert round(float(he["median_citation_density"]), 1) == 7.1


# --- 5. OR tables transcribed from the staged model outputs --------------------
def test_cumulative_or_table():
    cum = {(r["language"], r["topic"]): r for r in _read("cumulative_ors.csv")}
    expected = {
        ("ru", "g6pd"): (0.08, 0.00, 0.26, 99.8),
        ("ar", "g6pd"): (0.25, 0.01, 0.69, 98.1),
        ("he", "g6pd"): (0.51, 0.01, 1.61, 87.9),
        ("ru", "teratogenicity"): (0.40, 0.01, 1.26, 92.2),
        ("ar", "teratogenicity"): (0.69, 0.02, 2.18, 79.9),
        ("he", "teratogenicity"): (0.51, 0.01, 1.58, 88.0),
    }
    for key, exp in expected.items():
        r = cum[key]
        got = (round(float(r["mean"]), 2), round(float(r["hdi_low"]), 2),
               round(float(r["hdi_high"]), 2), round((1 - float(r["prob_or_gt_1"])) * 100, 1))
        assert got == exp, f"{key}: {got} != {exp}"


def test_hier_pooled_or_table():
    pooled = {r["language"]: r for r in _read("hier_pooled_ors.csv")}
    expected = {"he": (0.31, 0.01, 2.15), "ar": (0.25, 0.01, 2.10), "ru": (0.41, 0.01, 3.04)}
    for lang, exp in expected.items():
        r = pooled[lang]
        got = (round(float(r["or_median"]), 2), round(float(r["hdi_low"]), 2), round(float(r["hdi_high"]), 2))
        assert got == exp, f"{lang}: {got} != {exp}"


def test_hier_per_domain_or_table():
    per = {(r["language"], r["topic"]): r for r in _read("hier_per_topic_ors.csv")}
    expected = {
        "he": [0.33, 0.29, 0.09],
        "ar": [0.11, 0.38, 0.04],
        "ru": [0.05, 0.25, 1.06],
    }
    for lang, exp in expected.items():
        got = [round(float(per[(lang, t)]["or_median"]), 2)
               for t in ("g6pd", "teratogenicity", "qt")]
        assert got == exp, f"{lang}: {got} != {exp}"


# --- 6. warning_location figure counts -----------------------------------------
def test_warning_location_counts():
    GROUP = {"contraindications_section": "Contraindications",
             "adverse_effects_section": "Adverse effects", "side_effects": "Adverse effects",
             "pregnancy": "Adverse effects", "body": "Body", "lead": "Body",
             "footnote_only": "Body"}
    cats = ["Contraindications", "Adverse effects", "Body"]
    expected = {
        "g6pd_only": {"en": {"Contraindications": 4, "Adverse effects": 3, "Body": 0},
                      "ru": {"Contraindications": 2, "Adverse effects": 0, "Body": 0},
                      "ar": {"Contraindications": 1, "Adverse effects": 1, "Body": 2},
                      "he": {"Contraindications": 3, "Adverse effects": 1, "Body": 2}},
        "teratogen_only": {"en": {"Contraindications": 4, "Adverse effects": 4, "Body": 0},
                           "ru": {"Contraindications": 4, "Adverse effects": 0, "Body": 2},
                           "ar": {"Contraindications": 3, "Adverse effects": 0, "Body": 5},
                           "he": {"Contraindications": 4, "Adverse effects": 1, "Body": 2}},
    }
    files = {"g6pd_only": "g6pd_warnings.csv", "teratogen_only": "teratogenicity_warnings.csv"}
    for catg, fn in files.items():
        c = {l: {k: 0 for k in cats} for l in LANG_ORDER}
        for r in _read(fn):
            if r["category"] != catg or int(r["warning_code"]) <= 0:
                continue
            c[r["lang"]][GROUP.get(r["warning_location"], "Body")] += 1
        assert c == expected[catg], f"{catg}: {c}"


# --- 7. Canonical edition order ------------------------------------------------
def test_make_figures_edition_order_constants():
    src = (REPO / "paper" / "make_figures.py").read_text()
    assert 'LANG_ORDER = ["en", "ru", "ar", "he"]' in src
    assert 'NONEN_ORDER = ["ru", "ar", "he"]' in src


def test_structural_sort_is_canonical():
    rows = _read("structural_quality.csv")
    rows.sort(key=lambda r: float(r["median_word_count"]), reverse=True)
    assert [r["lang"] for r in rows] == LANG_ORDER
