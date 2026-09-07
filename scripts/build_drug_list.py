"""
build_drug_list.py — Drug list curation script for the Wikipedia drug-safety disparity study.

Produces data/drugs.csv with columns:
    drug, cpic_risk, pregnancy_risk, category, wikidata_qid,
    en_title, he_title, ar_title, ru_title

Usage:
    python scripts/build_drug_list.py
"""

import pathlib
import sys

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

# ---------------------------------------------------------------------------
# Hardcoded drug list with Wikidata QIDs
# QIDs verified via https://www.wikidata.org/w/api.php?action=wbsearchentities
# ---------------------------------------------------------------------------

DRUGS = [
    # (drug_name, cpic_risk, pregnancy_risk, category, wikidata_qid)
    # G6PD-only drugs (CPIC-classified)
    ("rasburicase",      "Contraindicated",             "", "g6pd_only",      "Q3088229"),
    ("methylene_blue",   "Contraindicated",             "", "g6pd_only",      "Q422134"),
    ("pegloticase",      "Contraindicated",             "", "g6pd_only",      "Q7160794"),
    ("dapsone",          "Avoid in deficient patients",  "", "g6pd_only",      "Q422226"),
    ("nitrofurantoin",   "Use with caution",             "", "g6pd_only",      "Q422669"),
    ("phenazopyridine",  "Use with caution",             "", "g6pd_only",      "Q3114549"),
    # Both G6PD + teratogenicity
    ("primaquine",       "Contraindicated",             "FDA contraindicated",  "both", "Q419834"),
    # Teratogenicity-only drugs
    ("isotretinoin",     "", "Contraindicated (iPLEDGE/REMS)",            "teratogen_only", "Q287029"),
    ("valproic_acid",    "", "Contraindicated (FDA black box)",           "teratogen_only", "Q240642"),
    ("methotrexate",     "", "Contraindicated (PLLR)",                    "teratogen_only", "Q422232"),
    ("warfarin",         "", "Contraindicated (1st tri; black box 2nd/3rd)", "teratogen_only", "Q113368879"),
    ("thalidomide",      "", "Contraindicated (REMS)",                    "teratogen_only", "Q203174"),
    # Negative controls (no G6PD interaction, no teratogenicity)
    ("amoxicillin",      "", "", "control", "Q201928"),
    ("atorvastatin",     "", "", "control", "Q668093"),
    ("omeprazole",       "", "", "control", "Q422210"),  # metformin replaced: see methodological_decisions.md
]

LANGUAGES = ["en", "he", "ar", "ru"]

# ---------------------------------------------------------------------------
# SPARQL query
# ---------------------------------------------------------------------------

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def build_sparql_query(qids: list[str]) -> str:
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?articleTitle ?lang WHERE {{
  VALUES ?item {{ {values} }}
  ?article schema:about ?item ;
           schema:inLanguage ?lang ;
           schema:name ?articleTitle .
  FILTER(?lang IN ("en", "he", "ar", "ru"))
  FILTER(CONTAINS(STR(?article), "wikipedia.org/wiki/"))
}}
"""
    return query


def fetch_sitelinks(qids: list[str]) -> dict[tuple[str, str], str]:
    """
    Returns a dict keyed by (qid, lang) -> article_title.
    """
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.addCustomHttpHeader(
        "User-Agent",
        "g6pd-disparity-study/1.0 (https://github.com/bgbg/pharma_g6pd; research)"
    )
    sparql.setQuery(build_sparql_query(qids))
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()

    sitelinks: dict[tuple[str, str], str] = {}
    for binding in results["results"]["bindings"]:
        # item URI is like http://www.wikidata.org/entity/Q415473
        item_uri = binding["item"]["value"]
        qid = item_uri.rsplit("/", 1)[-1]
        lang = binding["lang"]["value"]
        title = binding["articleTitle"]["value"]
        sitelinks[(qid, lang)] = title

    return sitelinks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    qids = [row[4] for row in DRUGS]

    print("Querying Wikidata SPARQL for sitelinks …")
    try:
        sitelinks = fetch_sitelinks(qids)
    except Exception as exc:
        print(f"SPARQL query failed: {exc}", file=sys.stderr)
        print("Continuing with empty sitelinks (all titles will be blank).", file=sys.stderr)
        sitelinks = {}

    # Build rows
    rows = []
    for drug_name, cpic_risk, pregnancy_risk, category, qid in DRUGS:
        row = {
            "drug": drug_name,
            "cpic_risk": cpic_risk,
            "pregnancy_risk": pregnancy_risk,
            "category": category,
            "wikidata_qid": qid,
        }
        for lang in LANGUAGES:
            col = f"{lang}_title"
            row[col] = sitelinks.get((qid, lang), "")
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "drug", "cpic_risk", "pregnancy_risk", "category", "wikidata_qid",
        "en_title", "he_title", "ar_title", "ru_title",
    ])

    # Save CSV
    out_path = pathlib.Path(__file__).parent.parent / "data" / "drugs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

    # Summary
    total_expected = len(DRUGS) * len(LANGUAGES)
    found = sum(
        1
        for _, _, _, _, qid in DRUGS
        for lang in LANGUAGES
        if sitelinks.get((qid, lang), "")
    )
    missing = total_expected - found
    print(f"\nSitelink summary: {found}/{total_expected} found, {missing} missing")
    for lang in LANGUAGES:
        lang_found = sum(
            1 for _, _, _, _, qid in DRUGS if sitelinks.get((qid, lang), "")
        )
        print(f"  {lang}: {lang_found}/{len(DRUGS)}")

    # Category summary
    print("\nCategory summary:")
    for cat in ["g6pd_only", "both", "teratogen_only", "control"]:
        count = sum(1 for row in DRUGS if row[3] == cat)
        print(f"  {cat}: {count}")

    print("\nFull drug table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
