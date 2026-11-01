"""Tests for scripts/build_validation_cases.py (#77): cell selection, rendering, per-slot shuffle, blindness."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_validation_cases as bvc  # noqa: E402

# Anything that would reveal an automated or reference code (word-level, case-insensitive).
FORBIDDEN = re.compile(
    r"consensus|\bllms?\b|gpt|gemini|grok|openrouter|model_codes|final_code|human_final|adjudicat|\bmodels?\b",
    re.I)
TEXT_FIELDS = {"wikitext", "sections", "infobox"}  # article content may contain any word

WIKITEXT = """{{Infobox drug
| name = Testdrug
| image = X.png
| pregnancy_AU = D
| routes = oral
| atc = A01
}}
'''Testdrug''' is a drug.<ref>{{cite web|url=http://x}} hidden-ref-text</ref> [[File:Pic.jpg|thumb|caption text]]

== Contraindications ==
Do not use in pregnancy.

== References ==
{{reflist}}
"""


def _write_inputs(tmp_path):
    drugs = pd.DataFrame([
        {"drug": "alpha_drug", "cpic_risk": "Contraindicated", "pregnancy_risk": "", "qt_risk": "", "pgx_risk": "", "en_title": "Alpha drug"},
        {"drug": "beta_drug", "cpic_risk": "", "pregnancy_risk": "", "qt_risk": "Known risk", "pgx_risk": "", "en_title": "Beta drug"},
        {"drug": "ctrl_drug", "cpic_risk": "", "pregnancy_risk": "", "qt_risk": "", "pgx_risk": "", "en_title": "Control"},
    ])
    drugs.to_csv(tmp_path / "drugs.csv", index=False)
    recs = []
    for drug in drugs["drug"]:
        for lang in bvc.LANGS:
            recs.append({"drug": drug, "lang": lang, "title": f"{drug} {lang}", "exists": True,
                         "wikitext": WIKITEXT, "last_edit_date": "2026-03-01"})
    for r in recs:
        if (r["drug"], r["lang"]) == ("beta_drug", "he"):
            r["exists"], r["wikitext"] = False, ""
    (tmp_path / "articles.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    return tmp_path / "articles.jsonl", tmp_path / "drugs.csv"


def test_select_cells_positives_with_article_only(tmp_path):
    cells = bvc.select_cells(*_write_inputs(tmp_path))
    ids = {c["cell_id"] for c in cells}
    assert len(ids) == len(cells) == 4 + 3  # alpha x4, beta x3 (no he article); control excluded
    assert "alpha_drug__en__g6pd" in ids and "beta_drug__ar__qt" in ids
    assert "beta_drug__he__qt" not in ids and not any(i.startswith("ctrl_drug") for i in ids)
    c = next(c for c in cells if c["cell_id"] == "alpha_drug__he__g6pd")
    assert c["drug"] == "Alpha drug" and c["domain"] == "g6pd"
    assert c["url"].startswith("https://he.wikipedia.org/wiki/")
    assert c["history_url"].endswith("&offset=20260301235959")


def test_render_article_drops_refs_and_files_keeps_infobox_and_headings():
    r = bvc.render_article(WIKITEXT)
    assert ["pregnancy_AU", "D"] in r["infobox"] and ["name", "Testdrug"] in r["infobox"]
    assert not any(k == "image" for k, _ in r["infobox"])
    body = " ".join(s["text"] for s in r["sections"])
    assert "hidden-ref-text" not in body and "caption text" not in body and "Pic.jpg" not in body
    assert "Do not use in pregnancy." in body
    assert [s["title"] for s in r["sections"] if s["title"]] == ["Contraindications"]


def test_per_slot_shuffle_differs_and_is_deterministic(tmp_path):
    ids = [f"cell{i:02d}" for i in range(40)]
    orders = {s: bvc.order_for_slot(ids, s) for s in bvc.SLOTS}
    assert all(sorted(o) == ids for o in orders.values())
    assert len({tuple(o) for o in orders.values()}) == len(bvc.SLOTS)
    assert orders["maya"] == bvc.order_for_slot(list(reversed(ids)), "maya")


def test_write_slots_names_and_blindness(tmp_path):
    cells = bvc.select_cells(*_write_inputs(tmp_path))
    out = tmp_path / "cases"
    bvc.write_slots(cells, out)
    for slot, name in bvc.SLOTS.items():
        written = json.loads((out / slot / "cases.json").read_text(encoding="utf-8"))
        assert {c["cell_id"] for c in written} == {c["cell_id"] for c in cells}
        assert (out / slot / "annotator_name.txt").read_text(encoding="utf-8").strip() == name
        for c in written:
            assert set(c) <= bvc.ALLOWED_KEYS
            meta = json.dumps({k: v for k, v in c.items() if k not in TEXT_FIELDS})
            assert not FORBIDDEN.search(meta), meta
    assert (out / "alpha" / "annotator_name.txt").read_text() == ""  # prompts for a name
    assert [c["cell_id"] for c in json.loads((out / "einat" / "cases.json").read_text())] != \
           [c["cell_id"] for c in json.loads((out / "maya" / "cases.json").read_text())]


def test_real_pool_is_24_drugs_and_blind():
    """The real pool built from the tracked snapshot: at most 96 cells, all metadata free of codes."""
    cells = bvc.select_cells(bvc.ARTICLES_JSONL, bvc.DRUGS_CSV)
    assert len({c["cell_id"].split("__")[0] for c in cells}) == 24
    assert 0 < len(cells) <= 96
    for c in cells:
        assert set(c) <= bvc.ALLOWED_KEYS
        assert not FORBIDDEN.search(json.dumps({k: v for k, v in c.items() if k not in TEXT_FIELDS}))
