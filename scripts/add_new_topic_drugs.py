"""
add_new_topic_drugs.py — Extend data/drugs.csv to four safety topics (M3).

Adds two new topic-membership columns (`qt_risk`, `pgx_risk`) and appends the
QT/Torsades and pharmacogenetic-SCAR drug rosters, while preserving the
committed 25-drug dataset exactly. Idempotent: re-running does not duplicate rows
and only fills blanks.

Topic membership is now expressed by the per-topic risk columns, not the single
`category` label, so a drug can be a positive for more than one topic:
    g6pd          -> cpic_risk
    teratogenicity-> pregnancy_risk
    qt            -> qt_risk
    pgx_scar      -> pgx_risk
A control is a drug with all four risk columns empty.

QIDs are resolved from the English Wikipedia title via the MediaWiki API
(robust against name collisions), then he/ar/ru/en titles are fetched from
Wikidata sitelinks via SPARQL.

Usage:
    conda run -n pharmaco python scripts/add_new_topic_drugs.py
"""

from __future__ import annotations

import sys
import time

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from g6pd.config import DRUGS_CSV  # noqa: E402

UA = "g6pd-disparity-study/1.0 (https://github.com/bgbg/pharma_g6pd; research)"
LANGUAGES = ["en", "he", "ar", "ru"]

# ---------------------------------------------------------------------------
# New rosters. Tuple: (drug_slug, en_title, qt_risk, pgx_risk)
# Gold standards: QT = CredibleMeds "Known Risk of TdP"; PGx = CPIC Level A.
# ---------------------------------------------------------------------------
QT_RISK = "Known risk of TdP (CredibleMeds)"

NEW_DRUGS = [
    # Topic 3 — QT prolongation / Torsades (CredibleMeds Known Risk)
    ("citalopram",         "Citalopram",         QT_RISK, ""),
    ("azithromycin",       "Azithromycin",       QT_RISK, ""),
    ("levofloxacin",       "Levofloxacin",       QT_RISK, ""),
    ("ondansetron",        "Ondansetron",        QT_RISK, ""),
    ("haloperidol",        "Haloperidol",        QT_RISK, ""),
    ("methadone",          "Methadone",          QT_RISK, ""),
    ("hydroxychloroquine", "Hydroxychloroquine", QT_RISK, ""),
    ("domperidone",        "Domperidone",        QT_RISK, ""),
    # Topic 4 — Pharmacogenetic severe ADR (CPIC): HLA -> SJS/TEN
    ("carbamazepine", "Carbamazepine", "", "HLA-B*15:02 SJS/TEN; HLA-A*31:01 (CPIC, FDA boxed)"),
    ("oxcarbazepine", "Oxcarbazepine", "", "HLA-B*15:02 SJS/TEN (CPIC)"),
    ("lamotrigine",   "Lamotrigine",   "", "HLA-B*15:02 SJS/TEN (CPIC)"),
    ("allopurinol",   "Allopurinol",   "", "HLA-B*58:01 SJS/TEN/DRESS (CPIC)"),
    ("abacavir",      "Abacavir",      "", "HLA-B*57:01 hypersensitivity (CPIC, FDA boxed, mandatory genotype)"),
    # Topic 4 — CYP2D6 -> opioid toxicity (ultrarapid metabolizers)
    ("codeine",  "Codeine",  "", "CYP2D6 UM respiratory depression (CPIC, FDA contraindicated children/breastfeeding)"),
    ("tramadol", "Tramadol", "", "CYP2D6 UM respiratory depression (CPIC)"),
]

# phenytoin is already in the dataset as a teratogen; mark it a dual-topic PGx
# positive (HLA-B*15:02 SJS/TEN + CYP2C9). Keyed by existing drug slug.
EXISTING_PGX_UPDATES = {
    "phenytoin": "HLA-B*15:02 SJS/TEN; CYP2C9 (CPIC, FDA boxed)",
}


def resolve_qid(en_title: str) -> str:
    """English Wikipedia title -> Wikidata QID via MediaWiki pageprops."""
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "titles": en_title,
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": 1,
            "format": "json",
        },
        headers={"User-Agent": UA},
        timeout=30,
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    qid = page.get("pageprops", {}).get("wikibase_item", "")
    if not qid:
        raise RuntimeError(f"No QID for {en_title!r} (page: {page.get('title')})")
    return qid


def fetch_sitelinks(qids: list[str]) -> dict[tuple[str, str], str]:
    """QID -> {(qid, lang): title} via the Wikidata wbgetentities API.

    Avoids the WDQS SPARQL endpoint (rate-limited during outages); the
    sitelinks REST API serves the same data in one batched call.
    """
    site_to_lang = {"enwiki": "en", "hewiki": "he", "arwiki": "ar", "ruwiki": "ru"}
    out: dict[tuple[str, str], str] = {}
    # wbgetentities accepts up to 50 ids per request.
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "sitelinks",
                "sitefilter": "|".join(site_to_lang),
                "format": "json",
            },
            headers={"User-Agent": UA},
            timeout=60,
        )
        resp.raise_for_status()
        entities = resp.json().get("entities", {})
        for qid, ent in entities.items():
            for site, link in ent.get("sitelinks", {}).items():
                lang = site_to_lang.get(site)
                if lang:
                    out[(qid, lang)] = link["title"]
    return out


def main() -> None:
    df = pd.read_csv(DRUGS_CSV).fillna("")
    for col in ("qt_risk", "pgx_risk"):
        if col not in df.columns:
            df[col] = ""

    # Mark dual-topic positives among existing drugs.
    for slug, pgx in EXISTING_PGX_UPDATES.items():
        mask = df["drug"] == slug
        if not mask.any():
            print(f"WARNING: existing drug {slug!r} not found; skipping PGx mark", file=sys.stderr)
        else:
            df.loc[mask, "pgx_risk"] = pgx
            print(f"Marked {slug} as dual-topic PGx positive.")

    existing = set(df["drug"])
    to_add = [d for d in NEW_DRUGS if d[0] not in existing]
    print(f"{len(to_add)} new drugs to add ({len(NEW_DRUGS) - len(to_add)} already present).")

    if to_add:
        qids: dict[str, str] = {}
        for slug, en_title, _, _ in to_add:
            qid = resolve_qid(en_title)
            qids[slug] = qid
            print(f"  {slug:20s} {en_title:20s} -> {qid}")
            time.sleep(0.2)

        sitelinks = fetch_sitelinks(list(qids.values()))

        rows = []
        for slug, en_title, qt_risk, pgx_risk in to_add:
            qid = qids[slug]
            # category retained for backward compatibility; topic membership now
            # comes from the risk columns. New positives labelled by lead topic.
            category = "qt_only" if qt_risk else "pgx_only"
            row = {
                "drug": slug,
                "cpic_risk": "",
                "pregnancy_risk": "",
                "category": category,
                "wikidata_qid": qid,
                "qt_risk": qt_risk,
                "pgx_risk": pgx_risk,
            }
            for lang in LANGUAGES:
                row[f"{lang}_title"] = sitelinks.get((qid, lang), "")
            rows.append(row)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    col_order = [
        "drug", "cpic_risk", "pregnancy_risk", "qt_risk", "pgx_risk",
        "category", "wikidata_qid", "en_title", "he_title", "ar_title", "ru_title",
    ]
    df = df[col_order]
    df.to_csv(DRUGS_CSV, index=False)

    print(f"\nSaved {len(df)} drugs to {DRUGS_CSV}")
    print("\nPer-topic positive counts (non-empty risk column):")
    for topic, col in [("g6pd", "cpic_risk"), ("teratogenicity", "pregnancy_risk"),
                       ("qt", "qt_risk"), ("pgx_scar", "pgx_risk")]:
        print(f"  {topic:15s} {int((df[col] != '').sum())}")
    controls = df[(df[["cpic_risk", "pregnancy_risk", "qt_risk", "pgx_risk"]] == "").all(axis=1)]
    print(f"  {'controls':15s} {len(controls)}  ({', '.join(controls['drug'])})")


if __name__ == "__main__":
    main()
