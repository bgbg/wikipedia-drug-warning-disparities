"""
build_validation_cases.py: write one blind-coding `cases.json` per annotator slot (#77).

Cells are the risk-positive (drug, language, domain) triples: the 24 drugs whose
drugs.csv risk column is non-empty for an analysed domain (g6pd, teratogenicity,
qt), crossed with the four languages, minus cells with no Wikipedia article.
Each slot gets its own seeded shuffle, so a partly finished slot is a random sample.

The output carries article text and display metadata ONLY. No LLM, model or
consensus code is read or written here (blindness invariant); the field
allow-list below is asserted at write time. Output is gitignored: raw Wikipedia
text stays out of the repo.

Usage:
    python scripts/build_validation_cases.py
    python scripts/build_validation_cases.py --articles /path/to/articles.jsonl

`--articles` defaults to ARTICLES_JSONL (src/g6pd/config.py); pass it to read a copy
kept elsewhere. (data/raw/articles.jsonl is tracked, so worktrees have it too.)
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from urllib.parse import quote

import mwparserfromhell as mw
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.coding import DISEASE_TERMS, HEMOLYSIS_TERMS  # noqa: E402
from g6pd.config import (  # noqa: E402
    ARTICLES_JSONL,
    DRUGS_CSV,
    TOPIC_WARNINGS,
    VALIDATION_CASES_DIR,
)

SHUFFLE_SEED = 20261002
LANGS = ["en", "he", "ar", "ru"]
# slot (URL path, systemd unit, data dir) -> name pre-seeded in annotator_name.txt.
# Alpha and Beta stay empty: the app asks for the name once.
SLOTS = {"einat": "Einat", "maya": "Maya", "nofar": "Nofar", "alpha": "", "beta": ""}
DOMAIN_LABELS = {"g6pd": "G6PD deficiency", "teratogenicity": "Teratogenicity (pregnancy)",
                 "qt": "QT prolongation / Torsades"}

ALLOWED_KEYS = {
    "cell_id", "drug", "lang", "domain", "title", "url", "history_url",
    "last_edit_date", "infobox", "sections", "wikitext", "terms",
}

# Highlight aids (substring, case-insensitive). "topic" marks mentions of the risk,
# "directive" marks avoid/contraindicated style wording. Aids only, never codes.
TOPIC_TERMS = {
    "teratogenicity": {
        "en": ["pregnan", "fetus", "fetal", "foetus", "foetal", "teratogen", "birth defect",
               "malformation", "congenital", "embryo", "childbearing", "contracepti"],
        "he": ["הריון", "היריון", "העובר", "בעובר", "עוברי", "טרטוגנ", "מומים", "הפלה"],
        "ar": ["الحمل", "حامل", "حوامل", "جنين", "ماسخ", "تشوه", "التشوه"],
        "ru": ["беременн", "плод", "тератоген", "порок", "врождённ", "врожденн", "эмбрио"],
    },
    "qt": {
        "en": ["qt", "torsade", "arrhythmi", "ventricular"],
        "he": ["qt", "טורסד", "הפרעות קצב", "הפרעת קצב", "קצב הלב"],
        "ar": ["qt", "تورساد", "التواء", "اضطراب", "نظم القلب", "تطاول"],
        "ru": ["qt", "torsade", "пируэт", "аритми", "удлинен", "желудочков"],
    },
}
DIRECTIVE_TERMS = {
    "en": ["contraindicat", "avoid", "do not", "must not", "should not", "not recommended",
           "not be used", "not be given", "ecg"],
    "he": ["התוויות נגד", "התווית נגד", "התוויית נגד", "להימנע", "אסור", "אין ליטול",
           "אין להשתמש", "אין לתת"],
    "ar": ["موانع", "مضاد استطباب", "مضادات الاستطباب", "يمنع", "يُمنع", "تجنب", "محظور",
           "لا يجوز", "لا ينبغي"],
    "ru": ["противопоказ", "не применя", "не следует", "избегать", "запрещ",
           "не рекомендуется", "не назнач"],
}

_FILE_NS = re.compile(r"^\s*(file|image|קובץ|תמונה|ملف|صورة|файл|изображение)\s*:", re.I)
_INFOBOX_SKIP = re.compile(r"^(image|caption|alt|width|imagesize|logo|upright)", re.I)


def topic_terms(domain: str, lang: str) -> list[str]:
    if domain == "g6pd":
        return list(dict.fromkeys(DISEASE_TERMS[lang] + HEMOLYSIS_TERMS[lang]))
    return TOPIC_TERMS[domain][lang]


def render_article(wikitext: str) -> dict:
    """Readable view of the wikitext: infobox key/values plus plain-text sections.

    Refs and file links are dropped; templates other than the leading infobox are
    dropped too (the raw wikitext stays available in the app for inspection).
    """
    code = mw.parse(wikitext)
    # strip_code keeps <ref> contents, so refs and file links are removed explicitly
    doomed = [t for t in code.filter_tags() if str(t.tag).strip().lower() == "ref"]
    doomed += [w for w in code.filter_wikilinks() if _FILE_NS.match(str(w.title))]
    for node in doomed:
        try:
            code.remove(node)
        except ValueError:  # already removed with an enclosing node
            pass
    infobox: list[list[str]] = []
    sections: list[dict] = []
    for i, sec in enumerate(code.get_sections(flat=True, include_lead=True, include_headings=True)):
        text = str(sec)
        heads = sec.filter_headings(recursive=False)
        level, title = 0, ""
        if heads and text.startswith(str(heads[0])):
            level, title = heads[0].level, str(heads[0].title).strip()
            text = text[len(str(heads[0])):]
        body = mw.parse(text)
        if i == 0:
            for t in body.filter_templates(recursive=False):
                if len(t.params) >= 5:
                    for p in t.params:
                        k = str(p.name).strip()
                        v = mw.parse(str(p.value)).strip_code().strip()
                        if v and not _INFOBOX_SKIP.match(k):
                            infobox.append([k, v[:300]])
        plain = body.strip_code(normalize=True, collapse=True).strip()
        if plain:
            sections.append({"level": level, "title": title, "text": plain})
    return {"infobox": infobox, "sections": sections}


def build_cell(rec: dict, domain: str, drug_label: str) -> dict:
    lang, title = rec["lang"], rec["title"]
    slug = quote(title.replace(" ", "_"), safe="")
    base = f"https://{lang}.wikipedia.org"
    edit = (rec.get("last_edit_date") or "").replace("-", "")
    case = {
        "cell_id": f"{rec['drug']}__{lang}__{domain}",
        "drug": drug_label,
        "lang": lang,
        "domain": domain,
        "title": title,
        "url": f"{base}/wiki/{slug}",
        # No revision id is stored with the snapshot; the history view anchored at the
        # last edit date lands on the archived revision (newest on that day first).
        "history_url": (f"{base}/w/index.php?title={slug}&action=history&offset={edit}235959"
                        if edit else ""),
        "last_edit_date": rec.get("last_edit_date") or "",
        "wikitext": rec["wikitext"],
        "terms": {"topic": topic_terms(domain, lang), "directive": DIRECTIVE_TERMS[lang]},
    }
    case.update(render_article(rec["wikitext"]))
    return case


def select_cells(articles_path: Path, drugs_path: Path) -> list[dict]:
    """All risk-positive cells with an existing article, in a canonical order."""
    drugs = pd.read_csv(drugs_path).fillna("")
    positives = []  # (drug, domain, label)
    for domain, (_csv, risk_col) in TOPIC_WARNINGS.items():
        for _, row in drugs[drugs[risk_col].astype(str).str.strip() != ""].iterrows():
            positives.append((row["drug"], domain, row["en_title"] or row["drug"]))
    wanted = {(d, dom): label for d, dom, label in positives}
    by_drug_lang = {}
    with open(articles_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                by_drug_lang[(rec["drug"], rec["lang"])] = rec
    cells = []
    for (drug, domain), label in sorted(wanted.items(), key=lambda x: (x[0][1], x[0][0])):
        for lang in LANGS:
            rec = by_drug_lang.get((drug, lang))
            if rec and rec.get("exists") and rec.get("wikitext"):
                cells.append(build_cell(rec, domain, label))
    return cells


def order_for_slot(cell_ids: list[str], slot: str) -> list[str]:
    """Deterministic per-slot shuffle (string seeds hash with sha512, stable across runs)."""
    ids = sorted(cell_ids)
    random.Random(f"{SHUFFLE_SEED}|{slot}").shuffle(ids)
    return ids


def write_slots(cells: list[dict], out_dir: Path, slots: dict[str, str] = SLOTS) -> None:
    by_id = {c["cell_id"]: c for c in cells}
    for c in cells:
        assert set(c) <= ALLOWED_KEYS, f"unexpected fields in case: {set(c) - ALLOWED_KEYS}"
    for slot, name in slots.items():
        d = out_dir / slot
        d.mkdir(parents=True, exist_ok=True)
        ordered = [by_id[i] for i in order_for_slot(list(by_id), slot)]
        (d / "cases.json").write_text(json.dumps(ordered, ensure_ascii=False), encoding="utf-8")
        (d / "annotator_name.txt").write_text((name + "\n") if name else "", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--articles", type=Path, default=ARTICLES_JSONL)
    ap.add_argument("--drugs", type=Path, default=DRUGS_CSV)
    ap.add_argument("--out", type=Path, default=VALIDATION_CASES_DIR)
    args = ap.parse_args()
    articles = args.articles
    if not articles.exists():
        sys.exit(f"articles file not found: {articles}")
    cells = select_cells(articles, args.drugs)
    write_slots(cells, args.out)
    per_lang = pd.Series([c["lang"] for c in cells]).value_counts().reindex(LANGS, fill_value=0)
    print(f"{len(cells)} cells from {articles}\n{per_lang.to_string()}")
    print(f"wrote {len(SLOTS)} slots under {args.out}")


if __name__ == "__main__":
    main()
