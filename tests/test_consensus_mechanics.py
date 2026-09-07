"""Regression tests locking the LLM consensus mechanics (audit issue #38).

No network. These re-derive the consensus from the frozen artifacts with an
implementation independent of scripts/classify_warnings.py and assert it matches
the recorded codes, plus the no-dropped-cells / non-existent-coded-0 invariants.
Once green, future edits cannot silently regress a verified result.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CODED = REPO / "data" / "coded"
PAPER_DATA = REPO / "paper" / "data"

CLASSIFICATIONS = CODED / "llm_classifications.csv"
RESPONSE_LOG = CODED / "llm_response_log.jsonl"
ARTICLES = REPO / "data" / "raw" / "articles.jsonl"
DRUGS = REPO / "data" / "drugs.csv"
FOUR_TOPIC = CODED / "coded_four_topic.csv"

# PGx-SCAR dropped as an analysed topic 2026-07-01 (see methodological_decisions.md).
WARNING_SOURCES = {
    "g6pd": PAPER_DATA / "g6pd_warnings.csv",
    "teratogenicity": PAPER_DATA / "teratogenicity_warnings.csv",
    "qt": CODED / "qt_warnings.csv",
}
RISK_COL = {
    "g6pd": "cpic_risk",
    "teratogenicity": "pregnancy_risk",
    "qt": "qt_risk",
}
ALL_RISK_COLS = ["cpic_risk", "pregnancy_risk", "qt_risk", "pgx_risk"]
LANGS = ["en", "he", "ar", "ru"]


# --- independent re-implementations -----------------------------------------

def _parse_code(raw):
    content = raw
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
    try:
        code = int(json.loads(content)["code"])
    except Exception:
        return None
    return code if code in (0, 1, 2) else None


def _agreement_rule(codes):
    valid = [c for c in codes if c is not None and c >= 0]
    if not valid:
        return False, -1
    if len(set(valid)) == 1:
        return True, valid[0]
    return False, Counter(valid).most_common(1)[0][0]


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    with open(CLASSIFICATIONS, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def log_index():
    by_key = defaultdict(list)
    with open(RESPONSE_LOG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_key[(rec["model"], rec["drug"], rec["lang"], rec["topic"])].append(rec)
    resolved = {}
    for key, recs in by_key.items():
        code = None
        for rec in recs:
            if "error" in rec:
                continue
            c = _parse_code(rec.get("raw_response", ""))
            if c is not None:
                code = c
        resolved[key] = code
    return resolved, set(by_key.keys())


@pytest.fixture(scope="module")
def existence():
    exist = {}
    with open(ARTICLES, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            exist[(rec["drug"], rec["lang"])] = rec.get("exists", False)
    return exist


def _applicable(topic):
    out = set()
    for r in csv.DictReader(open(DRUGS, encoding="utf-8")):
        positive = r.get(RISK_COL[topic], "").strip() != ""
        control = not any(r.get(c, "").strip() for c in ALL_RISK_COLS)
        if positive or control:
            out.add(r["drug"])
    return out


# --- tests -------------------------------------------------------------------

def test_per_model_codes_match_response_log(rows, log_index, existence):
    """Every recorded model code equals the last parseable log response;
    cells with no log entry are non-existent articles auto-coded 0/0/0."""
    log_codes, log_keys = log_index
    for r in rows:
        for i in (1, 2, 3):
            key = (r[f"model_{i}"], r["drug"], r["lang"], r["topic"])
            if key not in log_keys:
                assert existence.get((r["drug"], r["lang"])) is False, key
                assert r["consensus"] == "true" and r[f"code_{i}"] == "0", key
                continue
            assert log_codes[key] == int(r[f"code_{i}"]), key


def test_consensus_flag_and_final_code_rederive(rows):
    """The recorded consensus flag and final_code match the re-derived rule."""
    for r in rows:
        codes = [int(r[f"code_{i}"]) for i in (1, 2, 3)]
        is_cons, final = _agreement_rule(codes)
        assert is_cons == (r["consensus"] == "true"), (r["drug"], r["lang"], r["topic"], codes)
        assert final == int(r["final_code"]), (r["drug"], r["lang"], r["topic"], codes)


def test_model_set_per_domain(rows):
    """GPT-4o + Gemini-2.5-flash constant; Grok-3 for g6pd/teratogenicity,
    Grok-4.3 for qt/pgx_scar."""
    expected_m3 = {
        "g6pd": "x-ai/grok-3",
        "teratogenicity": "x-ai/grok-3",
        "qt": "x-ai/grok-4.3",
    }
    for r in rows:
        # llm_classifications.csv retains the raw four-topic classifications;
        # PGx-SCAR was dropped as an analysed topic 2026-07-01, so skip it here.
        if r["topic"] not in expected_m3:
            continue
        assert r["model_1"] == "openai/gpt-4o"
        assert r["model_2"] == "google/gemini-2.5-flash"
        assert r["model_3"] == expected_m3[r["topic"]], (r["topic"], r["model_3"])


def test_published_agreement_rate_124_of_136():
    """The manuscript's 124/136 (91.2%) overall agreement for the original two
    domains re-derives from the published model_codes column."""
    cons = total = 0
    for topic in ("g6pd", "teratogenicity"):
        for r in csv.DictReader(open(WARNING_SOURCES[topic], encoding="utf-8")):
            total += 1
            if _agreement_rule(list(json.loads(r["model_codes"]).values()))[0]:
                cons += 1
    assert (cons, total) == (124, 136)


def test_no_applicable_cell_dropped():
    """coded_four_topic has every applicable (drug, lang, topic) cell:
    17 applicable drugs x 4 langs x 3 topics = 204 (PGx-SCAR dropped 2026-07-01)."""
    present = set(
        (r["drug"], r["lang"], r["topic"])
        for r in csv.DictReader(open(FOUR_TOPIC, encoding="utf-8"))
    )
    for topic in RISK_COL:
        ap = _applicable(topic)
        assert len(ap) == 17, (topic, len(ap))
        for drug in ap:
            for lang in LANGS:
                assert (drug, lang, topic) in present, (drug, lang, topic)
    assert len(present) == 204


def test_four_topic_faithful_to_sources_and_nonexistent_coded_zero(existence):
    """coded_four_topic transcribes its source warning files exactly, and no
    non-existent article carries a non-zero code."""
    present = {
        (r["drug"], r["lang"], r["topic"]): int(r["final_code"])
        for r in csv.DictReader(open(FOUR_TOPIC, encoding="utf-8"))
    }
    for topic, path in WARNING_SOURCES.items():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            k = (r["drug"], r["lang"], topic)
            assert present[k] == int(r["warning_code"]), k
            if r["exists"] == "false":
                assert int(r["warning_code"]) == 0, k
