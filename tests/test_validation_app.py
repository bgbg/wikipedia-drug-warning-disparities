"""Tests for scripts/validation_app.py (#77): selftest, blindness of source and rendered pages.

The rendering tests need Streamlit (deploy/requirements.txt, not part of the analysis env,
whose pandas 3 conflicts with it) and are skipped when it is missing:
    python -m venv .venv-app && .venv-app/bin/pip install -r deploy/requirements.txt pytest pandas mwparserfromhell
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "scripts" / "validation_app.py"
sys.path.insert(0, str(REPO / "scripts"))

FORBIDDEN = re.compile(
    r"consensus|\bllms?\b|gpt|gemini|grok|openrouter|model_codes|final_code|human_final|adjudicat|\bmodels?\b",
    re.I)

CASES = [
    {"cell_id": f"drug{i}__{lang}__{dom}", "drug": f"Drug{i}", "lang": lang, "domain": dom,
     "title": f"Title{i}", "url": "https://x.org/a", "history_url": "https://x.org/h",
     "last_edit_date": "2026-01-01", "infobox": [["k", "v"]],
     "sections": [{"level": 0, "title": "", "text": "Avoid in pregnancy.\n\nSecond paragraph."}],
     "wikitext": "plain wikitext", "terms": {"topic": ["pregnan"], "directive": ["avoid"]}}
    for i, (lang, dom) in enumerate([("en", "teratogenicity"), ("he", "qt"), ("ar", "g6pd"), ("ru", "qt")])
]


def test_selftest():
    out = subprocess.run([sys.executable, str(APP), "--selftest"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "selftest OK" in out.stdout
    assert "skipped" not in out.stdout, "per-annotator shuffle check must run in the test environment"


def test_blind_app_source_has_no_model_or_consensus_wording():
    hits = FORBIDDEN.findall(APP.read_text(encoding="utf-8"))
    assert not hits, hits


def _app_test(tmp_path, monkeypatch, cases, name="Dana"):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "annotator_name.txt").write_text(name + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _chrome(at) -> str:
    """Everything the page shows outside the article body and the source expander."""
    parts = [m.value for m in at.markdown if not m.value.startswith(('<div class="art"', "<pre"))]
    parts += [e.value for e in at.sidebar.markdown] + [c.value for c in at.caption]
    parts += [b.label for b in at.button] + [s.label for s in at.sidebar.selectbox]
    parts += [t.label for t in at.text_area] + [e.label for e in at.expander]
    parts += [s.value for s in at.success] + [s.value for s in at.error]
    return "\n".join(parts)


def _click(at, code):
    at.button(key=f"b_{at.session_state['cur']}_{code}").click().run()
    assert not at.exception, at.exception


def test_blind_rendered_pages_with_clicks(tmp_path, monkeypatch):
    at = _app_test(tmp_path, monkeypatch, CASES)
    seen = [_chrome(at)]
    _click(at, "2")
    seen.append(_chrome(at))  # includes the "Saved to disk" banner and the next cell
    _click(at, "cant_tell")
    seen.append(_chrome(at))
    for page in seen:
        assert not FORBIDDEN.search(page), page
    assert (tmp_path / "annotations_Dana.csv").exists()
    rows = (tmp_path / "annotations_Dana.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 3 and rows[1].startswith("drug0__en__teratogenicity,Dana,2,")
    assert rows[2].split(",")[2] == "cant_tell"


def test_blind_real_cases_render_without_codes(tmp_path, monkeypatch):
    import build_validation_cases as bvc
    cells = bvc.select_cells(bvc.ARTICLES_JSONL, bvc.DRUGS_CSV)
    ordered = [c for c in cells if c["lang"] in ("he", "ar")][:3] + cells[:2]
    at = _app_test(tmp_path, monkeypatch, ordered)
    for _ in range(len(ordered)):
        assert not FORBIDDEN.search(_chrome(at)), _chrome(at)
        _click(at, "0")


def test_alpha_slot_asks_for_a_name(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    at = _app_test(tmp_path, monkeypatch, CASES, name="")
    assert [t.label for t in at.text_input] == ["Name"]
    at.text_input[0].set_value("Dana Lev")
    at.button[0].click().run()
    assert (tmp_path / "annotator_name.txt").read_text(encoding="utf-8").strip() == "Dana Lev"
    assert at.button(key=f"b_{at.session_state['cur']}_1")  # the coding screen now shows
