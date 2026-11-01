"""Blind validation UI for the Wikipedia drug-safety codes (#77).

One Streamlit process per annotator. Each process runs from its own working
directory, which holds `cases.json` (built by scripts/build_validation_cases.py),
`annotator_name.txt` and the annotator's own `annotations_<name>.csv`. The app
never receives, reads or shows any automated or reference code: the annotator
codes each article from its text alone.

Run:        streamlit run scripts/validation_app.py   (cwd = the annotator's directory)
Self-check: python scripts/validation_app.py --selftest

Deployed standalone on the VM: this file only needs the standard library and
Streamlit.
"""
from __future__ import annotations

import csv
import fcntl
import html
import io
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CASES_FILENAME = "cases.json"
NAME_FILENAME = "annotator_name.txt"
FIELDS = ["cell_id", "annotator", "code", "note", "saved_at"]
# Stored values. "cant_tell" is the CSV spelling of "can't tell".
CODES = ["0", "1", "2", "cant_tell"]
LANG_NAMES = {"en": "English", "he": "Hebrew", "ar": "Arabic", "ru": "Russian"}
RTL_LANGS = {"he", "ar"}
DOMAIN_NAMES = {
    "g6pd": "G6PD deficiency",
    "teratogenicity": "Teratogenicity (pregnancy)",
    "qt": "QT prolongation / Torsades",
}
# code -> (button label, keyboard shortcut). "C" is reserved by Streamlit, hence 3.
CODE_LABELS = {
    "0": ("0 · No mention", "0"),
    "1": ("1 · Mentioned, no instruction", "1"),
    "2": ("2 · Explicit avoid / contraindicated", "2"),
    "cant_tell": ("3 · Can't tell", "3"),
}

CODE_SHORT = {"0": "0", "1": "1", "2": "2", "cant_tell": "can't tell"}

RISK_PHRASE = {
    "g6pd": "G6PD (glucose-6-phosphate dehydrogenase) deficiency",
    "teratogenicity": "risk to the fetus in pregnancy",
    "qt": "QT prolongation / Torsades de Pointes (dangerous heart rhythm)",
}
GUIDE_COMMON = """
**Code only what the article text says.** Do not use your own medical knowledge about the drug:
if the article does not say it, it is a 0. Look in the infobox, the lead and every section.

* **0, No mention.** The risk is not mentioned anywhere in the article.
* **1, Mentioned, no instruction.** The risk is mentioned (for example in a pharmacology or side-effects
  section) but nothing tells the reader to avoid or not use the drug because of it.
* **2, Explicit avoid / contraindicated.** The article says the drug is contraindicated or must be
  avoided / not used because of the risk, including the risk being listed under a Contraindications heading.
* **Can't tell.** Use sparingly (unreadable or garbled text, or you genuinely cannot decide) and add a note.
"""
GUIDE_DOMAIN = {
    "g6pd": """
**G6PD.** Only G6PD / glucose-6-phosphate dehydrogenase deficiency counts. Hemolysis without any link to
G6PD is a 0. Phrases such as "contraindicated", "avoid", "do not use" or "should not be used" in connection
with G6PD make it a 2.
""",
    "teratogenicity": """
**Pregnancy.** Code what a non-specialist reader (a patient) would take from the article. A 2 needs a
directive a lay reader would understand as "do not take this drug if pregnant": contraindicated in
pregnancy, avoid in pregnancy, must not be used by pregnant women, women of childbearing potential must use
contraception.

* Also a **0**: an FDA pregnancy category A, B or C in the infobox with no pregnancy text; generic caution such
  as "use with caution in pregnancy", "consult your physician" or "use only if clearly needed".
* Also a **1**: birth defects or fetal harm described as fact (including historical accounts) without a
  directive to avoid the drug; FDA category D or X appearing only in the infobox.
* If the body text has an explicit directive, the body text decides, whatever the infobox says.
""",
    "qt": """
**QT / Torsades.** QT prolongation, the QT interval, Torsades de Pointes, or a drug-induced dangerous
ventricular arrhythmia. A 2 needs a directive: "contraindicated" in connection with QT prolongation, long-QT
syndrome or co-administration with other QT-prolonging drugs; the risk listed under Contraindications or
Warnings with an avoidance directive; instructions such as avoid, do not use, use only with ECG monitoring,
avoid in congenital long-QT or with low potassium or magnesium.
""",
}

# --- data layer (pure, unit-tested) ----------------------------------------

def _atomic_write(path: Path, content: str, *, encoding: str = "utf-8-sig") -> None:
    """Write `content` to `path` atomically: temp file, fsync, rename, fsync the directory.

    `utf-8-sig` (BOM) so Excel shows Hebrew and Arabic notes correctly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def safe_name(name: str) -> str:
    """File-safe form of an annotator name (letters, digits, underscore, hyphen)."""
    return re.sub(r"[^\w-]+", "_", name.strip()).strip("_")


def annotations_path(cases_dir: Path, annotator: str) -> Path:
    return cases_dir / f"annotations_{safe_name(annotator)}.csv"


def load_annotations(path: Path) -> dict[str, dict]:
    """cell_id -> full row (code may be empty when only a note was saved)."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["cell_id"]: r for r in csv.DictReader(f)}


def load_done(path: Path) -> dict[str, str]:
    """cell_id -> code for cells that carry a code."""
    return {k: r["code"] for k, r in load_annotations(path).items() if r.get("code")}


def save_annotation(path: Path, cell_id: str, annotator: str, *,
                    code: str | None = None, note: str | None = None) -> None:
    """Upsert one row by cell_id and rewrite the file atomically.

    `code` and `note` are saved independently: leaving one as None keeps the stored value.
    """
    if code is not None and code not in CODES:
        raise ValueError(f"invalid code {code!r}")
    # Streamlit re-executes this file on every run, so a module-level threading.Lock would not be
    # shared between sessions; an flock on a sidecar file serializes read-modify-write across tabs.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_annotations(path)
        row = rows.get(cell_id) or {"cell_id": cell_id, "code": "", "note": ""}
        if code is not None:
            row["code"] = code
        if note is not None:
            row["note"] = note
        row["annotator"] = annotator
        row["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows[cell_id] = row
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows.values())
        _atomic_write(path, buf.getvalue())


def filter_cases(cases: list[dict], done: dict[str, str], *, lang: str = "all",
                 domain: str = "all", unmarked_only: bool = False) -> list[dict]:
    return [c for c in cases
            if (lang == "all" or c["lang"] == lang)
            and (domain == "all" or c["domain"] == domain)
            and not (unmarked_only and c["cell_id"] in done)]


def first_unmarked(filtered: list[dict], done: dict[str, str]) -> int:
    """Index of the first unmarked cell in the filtered list; all marked: park on the last."""
    for i, c in enumerate(filtered):
        if c["cell_id"] not in done:
            return i
    return max(len(filtered) - 1, 0)


def neighbor_id(filtered: list[dict], cell_id: str) -> str:
    """The cell after `cell_id` in the filtered list (stays put on the last one)."""
    ids = [c["cell_id"] for c in filtered]
    if cell_id not in ids:
        return ids[0] if ids else cell_id
    return ids[min(ids.index(cell_id) + 1, len(ids) - 1)]


def language_counts(cases: list[dict], done: dict[str, str]) -> dict[str, tuple[int, int]]:
    """lang -> (marked, total) over all cells, ignoring filters."""
    out: dict[str, list[int]] = {}
    for c in cases:
        m = out.setdefault(c["lang"], [0, 0])
        m[1] += 1
        m[0] += c["cell_id"] in done
    return {k: (v[0], v[1]) for k, v in out.items()}


def find_cases_file() -> Path:
    for p in (Path.cwd() / CASES_FILENAME, Path(__file__).resolve().parent / CASES_FILENAME):
        if p.exists():
            return p
    raise FileNotFoundError("cases.json not found in the working directory")


def load_cases(path: Path) -> list[dict]:
    import json
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# --- rendering helpers (pure) ----------------------------------------------

def highlight(text: str, terms: dict) -> str:
    """HTML-escape `text` and wrap keyword hits in <mark> (aid only).

    terms = {"topic": [...], "directive": [...]}; directive wording wins on overlap.
    """
    esc = html.escape(text, quote=False)
    groups = []
    for name, key in (("d", "directive"), ("t", "topic")):
        ts = sorted({html.escape(t, quote=False) for t in terms.get(key, []) if t}, key=len, reverse=True)
        if ts:
            groups.append(f"(?P<{name}>" + "|".join(re.escape(t) for t in ts) + ")")
    if not groups:
        return esc
    pat = re.compile("|".join(groups), re.IGNORECASE)
    return pat.sub(lambda m: f'<mark class="{m.lastgroup}">{m.group(0)}</mark>', esc)


def article_html(case: dict) -> str:
    """Article body as one newline-free HTML block (blank lines would end a markdown HTML block)."""
    terms = case.get("terms", {})
    d = "rtl" if case["lang"] in RTL_LANGS else "ltr"
    parts = [f'<div class="art" dir="{d}">']
    if case.get("infobox"):
        rows = "".join(
            f'<tr><td class="k">{html.escape(k, quote=False)}</td><td dir="auto">{highlight(v, terms)}</td></tr>'
            for k, v in case["infobox"])
        parts.append(f'<div class="ibh">Infobox</div><table class="ib" dir="ltr">{rows}</table>')
    for sec in case.get("sections", []):
        if sec.get("title"):
            parts.append(f'<h4 class="sh">{html.escape(sec["title"], quote=False)}</h4>')
        for block in re.split(r"\n\s*\n", sec["text"]):
            block = block.strip()
            if block:
                parts.append("<p>" + highlight(block, terms).replace("\n", "<br>") + "</p>")
    parts.append("</div>")
    return "".join(parts)


CSS = """<style>
.art{background:#fff;color:#111;border:1px solid #ddd;border-radius:.5rem;padding:1rem 1.2rem;
     line-height:1.85;font-size:1.08rem;max-height:58vh;overflow-y:auto}
.art p{margin:.4rem 0}
.art h4.sh{margin:1rem 0 .2rem;font-size:1.05rem;color:#1a4a80;border-bottom:1px solid #e3e8ef}
.art mark{color:#111;padding:0 .1rem;border-radius:.15rem}
.art mark.t{background:#fff1a8}
.art mark.d{background:#ffcf99}
.art table.ib{border-collapse:collapse;font-size:.9rem;margin-bottom:.8rem;width:100%}
.art table.ib td{border:1px solid #e3e8ef;padding:.15rem .4rem;vertical-align:top}
.art table.ib td.k{color:#667;white-space:nowrap;direction:ltr;text-align:left}
.art .ibh{font-weight:700;color:#1a4a80;font-size:.85rem}
.meta{color:#666;font-size:.9rem}
.question{font-size:1.15rem;font-weight:600;margin:.2rem 0 .4rem}
.srcpre{direction:ltr;text-align:left;white-space:pre-wrap;font-size:.8rem;max-height:50vh;overflow-y:auto}
</style>"""



# --- UI --------------------------------------------------------------------

def _flash(kind: str, text: str) -> None:
    import streamlit as st
    st.session_state["flash"] = (kind, text)


def _note_changed(path: Path, cell_id: str, annotator: str, key: str) -> None:
    import streamlit as st
    try:
        save_annotation(path, cell_id, annotator, note=st.session_state[key])
        _flash("ok", "Note saved to disk.")
    except Exception as exc:  # shown to the annotator, never swallowed
        _flash("err", f"NOT SAVED: the note could not be written ({exc}). Tell the study team.")


def _ask_name(st, name_file: Path) -> None:
    st.markdown("### Enter your name to start")
    n = st.text_input("Name")
    if st.button("Start") and safe_name(n):
        _atomic_write(name_file, n.strip() + "\n", encoding="utf-8")
        st.rerun()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Drug-safety article validation", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        cases_path = find_cases_file()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    cases = load_cases(cases_path)
    cases_dir = cases_path.parent

    name_file = cases_dir / NAME_FILENAME
    annotator = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else ""
    if not annotator:
        _ask_name(st, name_file)
        return

    apath = annotations_path(cases_dir, annotator)
    done = load_done(apath)

    # sidebar: filters and progress per language
    st.sidebar.markdown(f"**Annotator:** {html.escape(annotator)}")
    langs_present = [x for x in LANG_NAMES if any(c["lang"] == x for c in cases)]
    lang = st.sidebar.selectbox("Language", ["all"] + langs_present,
                                format_func=lambda x: "All languages" if x == "all" else LANG_NAMES[x])
    domain = st.sidebar.selectbox("Domain", ["all"] + list(DOMAIN_NAMES),
                                  format_func=lambda x: "All domains" if x == "all" else DOMAIN_NAMES[x])
    unmarked_only = st.sidebar.toggle("Unmarked only", value=False)
    st.sidebar.markdown("**Progress**")
    counts = language_counts(cases, done)
    for lg in langs_present:
        m, t = counts.get(lg, (0, 0))
        st.sidebar.progress(m / t if t else 0.0, text=f"{LANG_NAMES[lg]}: {m} of {t}")
    st.sidebar.caption(f"Saved on disk: {len(done)} of {len(cases)} cells. "
                       "You may leave any language unmarked. Your marks are saved on every click.")

    filtered = filter_cases(cases, done, lang=lang, domain=domain, unmarked_only=unmarked_only)
    flash = st.session_state.pop("flash", None)
    if flash:
        (st.success if flash[0] == "ok" else st.error)(flash[1])
    if not filtered:
        st.info("No cells in this view. If 'Unmarked only' is on, every cell in the view is marked; "
                "turn it off to review or change marks.")
        return

    # resume at the first unmarked cell whenever the filter changes
    filt = (lang, domain, unmarked_only)
    ids = [c["cell_id"] for c in filtered]
    if st.session_state.get("filt") != filt or st.session_state.get("cur") not in ids:
        st.session_state["filt"] = filt
        st.session_state["cur"] = ids[first_unmarked(filtered, done)]
    cur = st.session_state["cur"]
    idx = ids.index(cur)
    case = filtered[idx]
    cid = case["cell_id"]
    row = load_annotations(apath).get(cid, {})
    current = row.get("code", "")

    drug, dom = case["drug"], case["domain"]
    st.markdown(f"<div class='meta'>Cell {idx + 1} of {len(filtered)} in this view &nbsp;|&nbsp; "
                f"<b>{html.escape(drug)}</b> &nbsp;|&nbsp; {LANG_NAMES[case['lang']]} Wikipedia &nbsp;|&nbsp; "
                f"{DOMAIN_NAMES[dom]}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='question'>What does the {LANG_NAMES[case['lang']]} article on "
                f"{html.escape(drug)} say about {RISK_PHRASE[dom]}?</div>", unsafe_allow_html=True)
    with st.expander("Coding guide", expanded=not done):
        st.markdown(GUIDE_COMMON + GUIDE_DOMAIN[dom])
        st.caption("Yellow highlights mark words about the risk, orange highlights mark avoid / "
                   "contraindicated style wording. They are only an aid: read the text.")

    links = [f"<a href='{html.escape(case['url'])}' target='_blank'>Live article</a>"]
    if case.get("history_url"):
        links.append(f"<a href='{html.escape(case['history_url'])}' target='_blank'>"
                     f"Archived revision (history as of {html.escape(case.get('last_edit_date', ''))})</a>")
    st.markdown("<div class='meta'>" + " &nbsp;|&nbsp; ".join(links)
                + f" &nbsp;|&nbsp; Title: {html.escape(case['title'])}</div>", unsafe_allow_html=True)
    st.markdown(article_html(case), unsafe_allow_html=True)
    with st.expander("Show source wikitext (everything the article contains, unformatted)"):
        st.markdown(f"<pre class='srcpre'>{html.escape(case['wikitext'], quote=False)}</pre>",
                    unsafe_allow_html=True)

    # the question: four buttons that save and advance
    cols = st.columns(4)
    for col, code in zip(cols, CODES):
        label, key = CODE_LABELS[code]
        mark = "✓ " if current == code else ""
        if col.button(mark + label, key=f"b_{cid}_{code}", shortcut=key, width="stretch", wrap=True,
                      type="primary" if current == code else "secondary"):
            try:
                save_annotation(apath, cid, annotator, code=code)
                if load_done(apath).get(cid) != code:
                    raise OSError("read-back check failed")
            except Exception as exc:
                _flash("err", f"NOT SAVED: {exc}. Do not continue; tell the study team.")
            else:
                _flash("ok", f"Saved to disk: {drug} ({LANG_NAMES[case['lang']]}) = "
                             f"{CODE_SHORT[code]}")
                st.session_state["cur"] = neighbor_id(filtered, cid)
            st.rerun()

    st.text_area("Note (optional; saved when you click outside the box)", value=row.get("note", ""),
                 key=f"note_{cid}", height=90, on_change=_note_changed,
                 args=(apath, cid, annotator, f"note_{cid}"))

    nav = st.columns(2)
    if nav[0].button("Previous", disabled=idx == 0, shortcut="Left", width="stretch"):
        st.session_state["cur"] = ids[idx - 1]
        st.rerun()
    if nav[1].button("Next", disabled=idx >= len(ids) - 1, shortcut="Right", width="stretch"):
        st.session_state["cur"] = ids[idx + 1]
        st.rerun()
    st.caption("Shortcuts: 0, 1, 2 = code, 3 = can't tell, left and right arrows = previous and next.")

    if len(done) == len(cases):
        st.success("Every cell is marked. Thank you!")


# --- self-check ------------------------------------------------------------

def _selftest() -> None:
    import json
    import subprocess

    cases = [{"cell_id": f"d{i}__{lg}__{dom}", "lang": lg, "domain": dom}
             for i, (lg, dom) in enumerate([("en", "qt"), ("he", "qt"), ("he", "g6pd"),
                                            ("ar", "qt"), ("he", "teratogenicity"), ("ru", "qt")])]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / CASES_FILENAME).write_text(json.dumps(cases), encoding="utf-8")
        ap = annotations_path(d, "Dana Lev")
        assert ap.name == "annotations_Dana_Lev.csv", ap.name

        # upsert: one row per cell, code and note saved independently
        assert load_done(ap) == {}
        save_annotation(ap, cases[0]["cell_id"], "Dana", code="1")
        assert load_done(ap) == {cases[0]["cell_id"]: "1"}
        save_annotation(ap, cases[0]["cell_id"], "Dana", code="2")
        assert load_done(ap) == {cases[0]["cell_id"]: "2"}
        assert len(load_annotations(ap)) == 1
        note = 'ללא אזהרה, "בעיה", שורה\nשנייה, with comma'
        save_annotation(ap, cases[0]["cell_id"], "Dana", note=note)
        row = load_annotations(ap)[cases[0]["cell_id"]]
        assert row["code"] == "2" and row["note"] == note, row
        save_annotation(ap, cases[1]["cell_id"], "Dana", note="only a note")
        assert cases[1]["cell_id"] not in load_done(ap)  # a note alone is not a mark
        save_annotation(ap, cases[1]["cell_id"], "Dana", code="cant_tell")
        row = load_annotations(ap)[cases[1]["cell_id"]]
        assert row["code"] == "cant_tell" and row["note"] == "only a note", row
        try:
            save_annotation(ap, cases[2]["cell_id"], "Dana", code="3")
            raise AssertionError("invalid code accepted")
        except ValueError:
            pass

        # atomic write: no stray files; a failed rename leaves the file intact and cleans up
        assert sorted(p.name for p in d.iterdir()) == sorted([CASES_FILENAME, ap.name, ap.name + ".lock"])
        before = ap.read_bytes()
        real_replace = os.replace
        os.replace = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            save_annotation(ap, cases[3]["cell_id"], "Dana", code="0")
            raise AssertionError("failed write not reported")
        except OSError:
            pass
        finally:
            os.replace = real_replace
        assert ap.read_bytes() == before
        assert sorted(p.name for p in d.iterdir()) == sorted([CASES_FILENAME, ap.name, ap.name + ".lock"])

        # resume at first unmarked within a filter; language filter and counts
        done = load_done(ap)  # marked: en/qt (d0), he/qt (d1)
        all_ = filter_cases(cases, done)
        assert first_unmarked(all_, done) == 2
        he = filter_cases(cases, done, lang="he")
        assert [c["cell_id"] for c in he][:1] == [cases[1]["cell_id"]] and len(he) == 3
        assert first_unmarked(he, done) == 1  # skips the marked he cell
        he_unmarked = filter_cases(cases, done, lang="he", unmarked_only=True)
        assert len(he_unmarked) == 2 and first_unmarked(he_unmarked, done) == 0
        assert len(filter_cases(cases, done, domain="qt")) == 4
        assert len(filter_cases(cases, done, lang="he", domain="qt", unmarked_only=True)) == 0
        assert language_counts(cases, done) == {"en": (1, 1), "he": (1, 3), "ar": (0, 1), "ru": (0, 1)}
        assert neighbor_id(he, he[0]["cell_id"]) == he[1]["cell_id"]
        assert neighbor_id(he, he[-1]["cell_id"]) == he[-1]["cell_id"]
        for c in cases:
            save_annotation(ap, c["cell_id"], "Dana", code="0")
        assert first_unmarked(all_, load_done(ap)) == len(all_) - 1  # all marked: park on last

        # highlight and article rendering escape HTML
        h = highlight("Avoid <b>QT</b> use", {"topic": ["qt"], "directive": ["avoid"]})
        assert '<mark class="d">Avoid</mark>' in h and '<mark class="t">QT</mark>' in h and "<b>" not in h, h

    # per-annotator shuffle (lives in the builder; skipped when this file is deployed alone)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_validation_cases import order_for_slot
    except ImportError:
        print("selftest: builder not importable, shuffle check skipped")
    else:
        ids = [f"cell{i:02d}" for i in range(30)]
        a, b = order_for_slot(ids, "einat"), order_for_slot(ids, "maya")
        assert sorted(a) == sorted(b) == ids
        assert a != b, "slots must get different orders"
        assert a == order_for_slot(list(reversed(ids)), "einat"), "order must not depend on input order"
        out = subprocess.run([sys.executable, "-c",
                              "import sys;sys.path.insert(0,sys.argv[1]);"
                              "from build_validation_cases import order_for_slot as o;"
                              "print(o([f'cell{i:02d}' for i in range(30)],'einat')[:3])",
                              str(Path(__file__).resolve().parent)],
                             capture_output=True, text=True, check=True).stdout.strip()
        assert out == str(a[:3]), "shuffle must be stable across processes"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
