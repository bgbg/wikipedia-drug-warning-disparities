"""SPARQL queries for the Wikidata structured data audit."""

import time
import logging

import requests
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WBGETENTITIES = "https://www.wikidata.org/w/api.php"
G6PD_QID = "Q848343"  # glucose-6-phosphate dehydrogenase deficiency
# NOTE (audit #37): the project previously used Q218093, which is "Zoch Verlag"
# (a board-game publisher), not G6PD deficiency. Corrected to Q848343. The audit
# headline (0/8 G6PD positives carry a P1909/P769 link) is unchanged: re-checking
# the eight G6PD drug QIDs against Q848343 also returns zero links.
_UA = "G6PDDisparityStudy/1.0 (https://github.com/bgbg/pharma_g6pd; boris@gorelik.net) research"

# Contraindication (P1909) and significant-drug-interaction (P769) are the two
# structured links the audit looks for between a drug and a safety condition.
CONTRAINDICATION_PROPS = ("P1909", "P769")


def check_drug_condition_links(
    qids: list[str],
    condition_qids: set[str],
    props: tuple[str, ...] = CONTRAINDICATION_PROPS,
) -> dict[str, list[str]]:
    """Which of ``props`` link each drug QID to any item in ``condition_qids``.

    Uses the wbgetentities claims REST API rather than the WDQS SPARQL endpoint,
    which is frequently rate-limited/outage-prone. Returns a dict mapping each
    QID that has at least one matching link to the list of property IDs found,
    e.g. ``{"Q419834": ["P1909"]}``; QIDs with no matching link are omitted.
    """
    result: dict[str, list[str]] = {}
    qids = [q for q in qids if q and q.startswith("Q")]
    for i in range(0, len(qids), 50):  # wbgetentities accepts <=50 ids/call
        chunk = qids[i:i + 50]
        for attempt in range(4):
            resp = requests.get(
                WBGETENTITIES,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "claims",
                    "format": "json",
                },
                headers={"User-Agent": _UA},
                timeout=60,
            )
            if resp.status_code in (429, 503):
                time.sleep(2.0 ** attempt)
                continue
            resp.raise_for_status()
            break
        entities = resp.json().get("entities", {})
        for qid, ent in entities.items():
            claims = ent.get("claims", {})
            for prop in props:
                for stmt in claims.get(prop, []):
                    snak = stmt.get("mainsnak", {})
                    val = snak.get("datavalue", {}).get("value", {})
                    target = val.get("id") if isinstance(val, dict) else None
                    if target in condition_qids:
                        result.setdefault(qid, [])
                        if prop not in result[qid]:
                            result[qid].append(prop)
    return result


def query_sparql(query: str) -> list[dict]:
    """Send a SPARQL query to Wikidata and return the result bindings.

    Retries once with a 5-second backoff on timeout or error.
    """
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.addCustomHttpHeader(
        "User-Agent",
        "pharma_g6pd_study/0.1 (research project)",
    )
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    for attempt in range(2):
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as exc:
            if attempt == 0:
                logger.warning(
                    "SPARQL query failed (%s); retrying in 5 s…", exc
                )
                time.sleep(5)
            else:
                logger.error("SPARQL query failed after retry: %s", exc)
                raise

    return []  # unreachable, but satisfies type checkers


def query_g6pd_linked_drugs() -> list[dict]:
    """Return all Wikidata drugs linked to G6PD deficiency (Q848343).

    Runs a UNION query covering P1909 (contraindication) and P769
    (significant drug interaction).

    Returns a list of dicts with keys:
        ``drug``      – full QID URI, e.g. "http://www.wikidata.org/entity/Q419834"
        ``drugLabel`` – English label string
    """
    query = f"""
SELECT DISTINCT ?drug ?drugLabel WHERE {{
  {{ ?drug wdt:P1909 wd:{G6PD_QID} }}
  UNION
  {{ ?drug wdt:P769 wd:{G6PD_QID} }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}
"""
    bindings = query_sparql(query)
    return [
        {
            "drug": b["drug"]["value"],
            "drugLabel": b.get("drugLabel", {}).get("value", ""),
        }
        for b in bindings
    ]


def check_drug_qids(qids: list[str]) -> dict[str, list[str]]:
    """Check which G6PD-deficiency properties each QID carries.

    Parameters
    ----------
    qids:
        List of QID strings including the "Q" prefix,
        e.g. ``["Q3088229", "Q419834"]``.

    Returns
    -------
    dict mapping each QID that has *any* G6PD link to the list of
    property IDs found, e.g. ``{"Q419834": ["P1909"]}``.
    QIDs with no G6PD link are omitted from the result.
    """
    values_clause = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?drug ?prop WHERE {{
  VALUES ?drug {{ {values_clause} }}
  VALUES ?prop {{ wdt:P1909 wdt:P769 }}
  ?drug ?prop wd:{G6PD_QID} .
}}
"""
    bindings = query_sparql(query)

    result: dict[str, list[str]] = {}
    for b in bindings:
        # drug URI → QID, e.g. "http://www.wikidata.org/entity/Q419834" → "Q419834"
        drug_uri = b["drug"]["value"]
        qid = drug_uri.rsplit("/", 1)[-1]
        # prop URI → property ID, e.g. ".../P1909" → "P1909"
        prop_uri = b["prop"]["value"]
        prop_id = prop_uri.rsplit("/", 1)[-1]
        result.setdefault(qid, [])
        if prop_id not in result[qid]:
            result[qid].append(prop_id)

    return result
