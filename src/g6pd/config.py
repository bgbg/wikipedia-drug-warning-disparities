from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CODED_DIR = DATA_DIR / "coded"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DRUGS_CSV = DATA_DIR / "drugs.csv"
ARTICLES_JSONL = RAW_DIR / "articles.jsonl"
WARNINGS_CSV = CODED_DIR / "warnings.csv"
WIKIDATA_GAPS_CSV = DATA_DIR / "wikidata_gaps.csv"
WIKIDATA_PROPERTIES_JSON = DATA_DIR / "wikidata_properties.json"
WIKIDATA_BASE_RATES_CSV = DATA_DIR / "wikidata_base_rates.csv"

# LLM classification pipeline
PROMPTS_DIR = DATA_DIR / "prompts"
CLASSIFICATION_PROMPTS = PROMPTS_DIR / "classification_prompts.json"
LLM_CLASSIFICATIONS_CSV = CODED_DIR / "llm_classifications.csv"
LLM_DISAGREEMENTS_CSV = CODED_DIR / "llm_disagreements.csv"
LLM_RESPONSE_LOG = CODED_DIR / "llm_response_log.jsonl"
G6PD_WARNINGS_CSV = CODED_DIR / "g6pd_warnings.csv"
TERATOGENICITY_WARNINGS_CSV = CODED_DIR / "teratogenicity_warnings.csv"
QT_WARNINGS_CSV = CODED_DIR / "qt_warnings.csv"
# Retained for provenance only: PGx-SCAR was piloted in M3 but DROPPED as an
# analysed topic on 2026-07-01 (Maya Berlin + team verdict; see
# methodological_decisions.md). The file/column stay as a historical record.
PGX_WARNINGS_CSV = CODED_DIR / "pgx_scar_warnings.csv"

# Analysed topics -> (per-topic warnings CSV, the drugs.csv risk column marking a
# positive). PGx-SCAR removed 2026-07-01; the study now covers three topics.
TOPIC_WARNINGS = {
    "g6pd": (G6PD_WARNINGS_CSV, "cpic_risk"),
    "teratogenicity": (TERATOGENICITY_WARNINGS_CSV, "pregnancy_risk"),
    "qt": (QT_WARNINGS_CSV, "qt_risk"),
}

# Control-defining risk columns: a drug is a clean negative control only if ALL
# of these are empty. `pgx_risk` stays here even though PGx-SCAR is no longer an
# analysed topic, so the 7 PGx-only drugs are EXCLUDED from the study rather than
# miscounted as negative controls.
CONTROL_RISK_COLS = ["cpic_risk", "pregnancy_risk", "qt_risk", "pgx_risk"]
METHOD_COMPARISON_CSV = CODED_DIR / "g6pd_method_comparison.csv"

# Blind validation of the LLM codes by the collaborators (#77). Per-slot cases
# (article text) and pulled annotation CSVs are gitignored; only the aggregated
# precheck/agreement tables directly under VALIDATION_DIR are committed.
CODED_FOUR_TOPIC_CSV = CODED_DIR / "coded_four_topic.csv"
VALIDATION_DIR = DATA_DIR / "validation"
VALIDATION_CASES_DIR = VALIDATION_DIR / "cases"
VALIDATION_ANNOTATIONS_DIR = VALIDATION_DIR / "annotations"
VALIDATION_BACKUPS_DIR = VALIDATION_DIR / "backups"

