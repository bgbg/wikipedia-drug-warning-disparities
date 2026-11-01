"""Wikidata structured-data audit: property lookups and SPARQL builders.

Property labels were re-verified with ``wbgetentities`` on 2/10/2026 and are
stored in ``data/wikidata_properties.json``:

    P10946  "contraindicated in case of"  (drug -> condition; the contraindication property)
    P3489   "pregnancy category"          (drug -> category; teratogenicity coverage)
    P1909   "side effect"                 (NOT a contraindication; reported as context only)
    P769    "significant drug interaction" (links two drugs, so it cannot point to a
                                            condition; reported as context only)
    P267    "ATC code"                    (defines the all-Wikidata base-rate universe)

Earlier versions of this module treated P1909 as "contraindication", which was
wrong (issue #76). Condition QID history: the project once used Q218093 for G6PD
deficiency, which is "Zoch Verlag"; the correct item is Q848343 (audit #37).
"""

import logging
import re
import time

import requests
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WBGETENTITIES = "https://www.wikidata.org/w/api.php"
_UA = "G6PDDisparityStudy/1.0 (https://github.com/bgbg/pharma_g6pd; boris@gorelik.net) research"

PROPERTY_IDS = ("P1909", "P769", "P3489", "P10946")  # labels saved to data/wikidata_properties.json
ATC_PROP = "P267"
CONTEXT_PROPS = ("P1909", "P769")  # reported beside the primary properties, never counted

# Safety-condition items per topic.
TOPIC_CONDITIONS = {
    "g6pd": {"Q848343"},  # glucose-6-phosphate dehydrogenase deficiency
    "teratogenicity": {"Q11995"},  # pregnancy
    "qt": {"Q653924", "Q1625433"},  # long QT syndrome, torsades de pointes
}


def topic_checks(topic: str) -> list[tuple[str, str, set[str] | None]]:
    """(property, role, target QIDs) triples audited for ``topic``.

    ``targets`` of None means any value of the property counts (P3489 records a
    pregnancy category on the drug itself, so there is no condition target).
    Role is "primary" (counted) or "context" (reported only).
    """
    cond = TOPIC_CONDITIONS[topic]
    primary: list[tuple[str, set[str] | None]] = [("P10946", cond)]
    if topic == "teratogenicity":
        primary.append(("P3489", None))
    return [(p, "primary", t) for p, t in primary] + [(p, "context", cond) for p in CONTEXT_PROPS]


def in_denominator(topic: str, risk_basis: str) -> bool:
    """Denominator rule (PENDING MAYA CONFIRMATION, issue #76).

    A positive counts toward the contraindication-link rate only if its reference
    standard says "contraindicated" (G6PD and teratogenicity). QT positives use
    "Known Risk of TdP", which is not a contraindication, so they are reported
    separately.
    """
    return topic in ("g6pd", "teratogenicity") and risk_basis.strip().lower().startswith("contraindicated")


def _get_json(params: dict) -> dict:
    """GET the Wikidata API with the project User-Agent; retry on 429/503.

    Raises on HTTP errors and on an ``error`` object in a 200 response body, so a
    failed fetch can never be mistaken for "no statements".
    """
    for attempt in range(4):
        resp = requests.get(WBGETENTITIES, params={**params, "format": "json"},
                            headers={"User-Agent": _UA}, timeout=60)
        if resp.status_code in (429, 503):
            time.sleep(2.0 ** attempt)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Wikidata API error: {data['error']}")
        return data
    resp.raise_for_status()
    raise RuntimeError("Wikidata API kept returning 429/503")


def _entities(data: dict, requested: list[str]) -> dict:
    """Entities of a ``wbgetentities`` response; raise if any requested ID is missing."""
    ents = data.get("entities", {})
    bad = [q for q in requested if q not in ents or "missing" in ents[q]]
    if bad:
        raise RuntimeError(f"Wikidata returned no entity for {bad}")
    return ents


def fetch_labels(ids: list[str], lang: str = "en") -> dict[str, dict]:
    """Label, description and datatype for each entity ID (``wbgetentities``)."""
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        data = _get_json({"action": "wbgetentities", "ids": "|".join(chunk),
                          "props": "labels|descriptions|datatype", "languages": lang})
        for eid, ent in _entities(data, chunk).items():
            out[eid] = {
                "label": ent.get("labels", {}).get(lang, {}).get("value", ""),
                "description": ent.get("descriptions", {}).get(lang, {}).get("value", ""),
                "datatype": ent.get("datatype", ""),
            }
    return out


def claim_values(claims: dict, props: tuple[str, ...]) -> dict[str, list[str]]:
    """{property: [values]} from a ``claims`` block, approximating ``wdt:`` truthy paths.

    Per property only best-rank statements count: the preferred ones if any exist,
    otherwise the normal ones; deprecated statements are skipped. Statements without
    a value (no value / unknown value) are skipped. This is an approximation of WDQS,
    not a full reimplementation. Item values are returned as QIDs, string values
    (e.g. an ATC code) as the string.
    """
    out: dict[str, list[str]] = {}
    for prop in props:
        stmts = [st for st in claims.get(prop, []) if st.get("rank") != "deprecated"]
        if any(st.get("rank") == "preferred" for st in stmts):
            stmts = [st for st in stmts if st.get("rank") == "preferred"]
        for stmt in stmts:
            val = stmt.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(val, dict):
                val = val.get("id")
            if val:
                out.setdefault(prop, []).append(val)
    return out


def fetch_drug_claims(qids: list[str], props: tuple[str, ...]) -> dict[str, dict[str, list[str]]]:
    """{drug QID: {property: [values]}} via ``wbgetentities`` (robust to WDQS outages)."""
    qids = [q for q in qids if q and q.startswith("Q")]
    out: dict[str, dict[str, list[str]]] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        data = _get_json({"action": "wbgetentities", "ids": "|".join(chunk), "props": "claims"})
        for qid, ent in _entities(data, chunk).items():
            out[qid] = claim_values(ent.get("claims", {}), props)
    return out


def matching_values(values: list[str], targets: set[str] | None) -> list[str]:
    """Values that satisfy a check: all of them if ``targets`` is None, else the overlap."""
    return list(values) if targets is None else [v for v in values if v in targets]


_PID = re.compile(r"P\d+")
_QID = re.compile(r"Q\d+")


def build_atc_count_query(prop: str | None = None, targets: set[str] | None = None) -> str:
    """SPARQL counting distinct ATC-coded (P267) items.

    With ``prop``, count only items that also carry ``prop`` (to any of ``targets``
    when given). Truthy ``wdt:`` paths, so deprecated statements are excluded.
    """
    body = f"?s wdt:{ATC_PROP} ?atc ."
    if prop:
        if not _PID.fullmatch(prop):
            raise ValueError(f"bad property id {prop!r}")
        body += f" ?s wdt:{prop} ?o ."
        if targets:
            if not all(_QID.fullmatch(t) for t in targets):
                raise ValueError(f"bad target ids {targets!r}")
            body += " VALUES ?o { " + " ".join(f"wd:{t}" for t in sorted(targets)) + " }"
    return f"SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE {{ {body} }}"


def build_statement_count_query(prop: str) -> str:
    """SPARQL counting every ``p:`` statement of ``prop`` on all of Wikidata.

    Unit: statements (including any deprecated ones), not distinct items.
    """
    if not _PID.fullmatch(prop):
        raise ValueError(f"bad property id {prop!r}")
    return f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s p:{prop} ?st . }}"


def query_sparql(query: str, timeout: int = 120) -> list[dict]:
    """Send a SPARQL query to Wikidata and return the result bindings.

    Retries once with a 5-second backoff on timeout or error.
    """
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.addCustomHttpHeader("User-Agent", _UA)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)

    for attempt in range(2):
        try:
            return sparql.query().convert()["results"]["bindings"]
        except Exception as exc:
            if attempt == 0:
                logger.warning("SPARQL query failed (%s); retrying in 5 s", exc)
                time.sleep(5)
            else:
                logger.error("SPARQL query failed after retry: %s", exc)
                raise
    return []  # unreachable, but satisfies type checkers


def count_atc_items(prop: str | None = None, targets: set[str] | None = None) -> int:
    """Run ``build_atc_count_query`` and return the count."""
    return int(query_sparql(build_atc_count_query(prop, targets))[0]["n"]["value"])


def count_statements(prop: str) -> int:
    """Run ``build_statement_count_query`` and return the count."""
    return int(query_sparql(build_statement_count_query(prop))[0]["n"]["value"])
