"""
classify_warnings.py — LLM-based warning classification pipeline.

Sends Wikipedia article text to three LLM models via the OpenRouter API
and determines warning presence (0/1/2) using a consensus protocol.

Usage:
    python scripts/classify_warnings.py --topic g6pd
    python scripts/classify_warnings.py --topic teratogenicity
    python scripts/classify_warnings.py --topic g6pd --force
    python scripts/classify_warnings.py --topic g6pd --models openai/gpt-4o,google/gemini-1.5-pro,x-ai/grok-2
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from g6pd.config import (
    ARTICLES_JSONL,
    CLASSIFICATION_PROMPTS,
    CODED_DIR,
    DRUGS_CSV,
    LLM_CLASSIFICATIONS_CSV,
    LLM_DISAGREEMENTS_CSV,
    LLM_RESPONSE_LOG,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# grok-3 was deprecated on OpenRouter (404) during M3; xAI's recommended
# successor grok-4.3 replaces it as the third consensus model for the new topics.
DEFAULT_MODELS = ["openai/gpt-4o", "google/gemini-2.5-flash", "x-ai/grok-4.3"]

# Topic -> the drugs.csv risk column that marks a positive for that topic.
# A drug is applicable to a topic if its risk column is non-empty (positive) or
# it is a negative control (all four risk columns empty). Selecting by risk
# column rather than the single `category` label lets a drug be a positive for
# more than one topic (e.g. phenytoin: teratogen + PGx-SCAR).
# PGx-SCAR dropped as an analysed topic 2026-07-01 (see methodological_decisions.md);
# phenytoin is retained via its teratogen (pregnancy_risk) membership.
TOPIC_RISK_COL = {
    "g6pd": "cpic_risk",
    "teratogenicity": "pregnancy_risk",
    "qt": "qt_risk",
}
# Control-defining columns: keep pgx_risk here so PGx-only drugs are EXCLUDED (not
# treated as negative controls) even though PGx-SCAR is no longer classified.
ALL_RISK_COLS = ["cpic_risk", "pregnancy_risk", "qt_risk", "pgx_risk"]

CLASSIFICATIONS_COLUMNS = [
    "drug", "lang", "topic",
    "model_1", "code_1",
    "model_2", "code_2",
    "model_3", "code_3",
    "consensus", "final_code", "location", "timestamp",
    "human_final_code", "adjudicator",
]

DISAGREEMENTS_COLUMNS = [
    "drug", "lang", "topic",
    "model_1", "code_1", "justification_1",
    "model_2", "code_2", "justification_2",
    "model_3", "code_3", "justification_3",
    "human_code", "human_reviewer",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_applicable_articles(topic: str) -> list[dict]:
    """Load articles applicable to the given topic based on drug category."""
    import pandas as pd

    if not ARTICLES_JSONL.exists():
        logger.error("Articles file not found: %s. Run collect_articles.py first.", ARTICLES_JSONL)
        sys.exit(1)

    drugs_df = pd.read_csv(DRUGS_CSV).fillna("")
    for col in ALL_RISK_COLS:
        if col not in drugs_df.columns:
            drugs_df[col] = ""
    risk_col = TOPIC_RISK_COL[topic]
    is_positive = drugs_df[risk_col].astype(str).str.strip() != ""
    is_control = (drugs_df[ALL_RISK_COLS].astype(str).apply(lambda s: s.str.strip()) == "").all(axis=1)
    applicable_drugs = set(drugs_df[is_positive | is_control]["drug"])

    articles = []
    with open(ARTICLES_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["drug"] in applicable_drugs:
                articles.append(rec)

    logger.info(
        "Loaded %d articles for topic '%s' (%d applicable drugs)",
        len(articles), topic, len(applicable_drugs),
    )
    return articles


def load_prompts(topic: str) -> dict:
    """Load system prompt and few-shot examples for a topic."""
    with open(CLASSIFICATION_PROMPTS, encoding="utf-8") as fh:
        all_prompts = json.load(fh)
    config = all_prompts[topic]
    # Support both "system" (single string) and "system_parts" (list of lines)
    if "system_parts" in config and "system" not in config:
        config["system"] = "\n".join(config["system_parts"])
    return config


def load_existing_classifications() -> set[tuple[str, str, str]]:
    """Load already-classified (drug, lang, topic) triples."""
    existing = set()
    if LLM_CLASSIFICATIONS_CSV.exists():
        with open(LLM_CLASSIFICATIONS_CSV, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing.add((row["drug"], row["lang"], row["topic"]))
    return existing


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def build_messages(prompt_config: dict, article_text: str, topic: str) -> list[dict]:
    """Build the message list for the API call."""
    messages = [{"role": "system", "content": prompt_config["system"]}]
    for example in prompt_config["examples"]:
        messages.append({"role": example["role"], "content": example["content"]})

    topic_label = {
        "g6pd": "G6PD warnings",
        "teratogenicity": "teratogenicity warnings",
        "qt": "QT-prolongation / Torsades de Pointes warnings",
    }[topic]
    user_msg = f"Classify this article for {topic_label}:\n\n{article_text}"
    messages.append({"role": "user", "content": user_msg})
    return messages


def classify_with_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    drug: str,
    lang: str,
    topic: str,
) -> dict:
    """Send classification request to one model. Returns parsed result or error."""
    timestamp = datetime.now(timezone.utc).isoformat()

    for attempt in range(2):  # max 1 retry
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=512,
            )
            raw_content = response.choices[0].message.content.strip()

            # Log the full response
            log_entry = {
                "model": model,
                "drug": drug,
                "lang": lang,
                "topic": topic,
                "timestamp": timestamp,
                "attempt": attempt + 1,
                "raw_response": raw_content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                },
            }
            with open(LLM_RESPONSE_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            # Parse JSON from response (handle markdown code blocks)
            content = raw_content
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            result = json.loads(content)

            code = int(result["code"])
            if code not in (0, 1, 2):
                raise ValueError(f"Invalid code: {code}")

            return {
                "code": code,
                "location": result.get("location", "absent"),
                "justification": result.get("justification", ""),
                "error": None,
            }

        except Exception as exc:
            if attempt == 0:
                logger.warning(
                    "  Retry %s for %s/%s/%s: %s", model, drug, lang, topic, exc
                )
                time.sleep(2)
                continue
            logger.error(
                "  Failed %s for %s/%s/%s after retry: %s",
                model, drug, lang, topic, exc,
            )
            # Log the failure
            log_entry = {
                "model": model,
                "drug": drug,
                "lang": lang,
                "topic": topic,
                "timestamp": timestamp,
                "attempt": attempt + 1,
                "error": str(exc),
            }
            with open(LLM_RESPONSE_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            return {
                "code": -1,
                "location": "absent",
                "justification": f"ERROR: {exc}",
                "error": str(exc),
            }

    # Should not reach here, but just in case
    return {"code": -1, "location": "absent", "justification": "ERROR: unknown", "error": "unknown"}


# ---------------------------------------------------------------------------
# Consensus logic
# ---------------------------------------------------------------------------

def determine_consensus(results: list[dict]) -> tuple[bool, int, str]:
    """Determine consensus from 3 model results.

    Returns (is_consensus, final_code, location).
    """
    codes = [r["code"] for r in results]
    valid_codes = [c for c in codes if c >= 0]

    if not valid_codes:
        return False, -1, "absent"

    from collections import Counter
    counter = Counter(valid_codes)
    most_common_code, count = counter.most_common(1)[0]
    locations = [r["location"] for r in results if r["code"] == most_common_code]
    location = next((loc for loc in locations if loc != "absent"), locations[0] if locations else "absent")

    # Consensus if all valid models agree (allows 2/3 agreement when 1 model errors)
    if len(set(valid_codes)) == 1:
        return True, valid_codes[0], location

    return False, most_common_code, location


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def ensure_csv_headers(path: Path, columns: list[str]) -> None:
    """Create CSV with headers if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)


def append_classification(row: dict) -> None:
    """Append a row to the classifications CSV."""
    with open(LLM_CLASSIFICATIONS_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLASSIFICATIONS_COLUMNS)
        writer.writerow(row)


def append_disagreement(row: dict) -> None:
    """Append a row to the disagreements CSV."""
    with open(LLM_DISAGREEMENTS_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DISAGREEMENTS_COLUMNS)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_classification(topic: str, models: list[str], force: bool) -> None:
    """Run the full classification pipeline for a topic."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment. Set it in .env")
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    articles = load_applicable_articles(topic)
    prompt_config = load_prompts(topic)
    existing = load_existing_classifications() if not force else set()

    # Ensure output files exist with headers
    CODED_DIR.mkdir(parents=True, exist_ok=True)
    ensure_csv_headers(LLM_CLASSIFICATIONS_CSV, CLASSIFICATIONS_COLUMNS)
    ensure_csv_headers(LLM_DISAGREEMENTS_CSV, DISAGREEMENTS_COLUMNS)

    n_classified = 0
    n_skipped = 0
    n_consensus = 0
    n_disagreement = 0
    n_auto_zero = 0

    for i, article in enumerate(articles):
        drug = article["drug"]
        lang = article["lang"]
        exists = article.get("exists", False)

        # Idempotency check
        if (drug, lang, topic) in existing:
            n_skipped += 1
            continue

        timestamp = datetime.now(timezone.utc).isoformat()

        # Non-existent articles get code 0 automatically
        if not exists:
            row = {
                "drug": drug, "lang": lang, "topic": topic,
                "model_1": models[0], "code_1": 0,
                "model_2": models[1], "code_2": 0,
                "model_3": models[2], "code_3": 0,
                "consensus": "true", "final_code": 0,
                "location": "absent", "timestamp": timestamp,
            }
            append_classification(row)
            n_auto_zero += 1
            n_classified += 1
            logger.info("[%d/%d] %s/%s — article missing, auto-coded 0", i + 1, len(articles), drug, lang)
            continue

        # Build prompt with article text
        wikitext = article.get("wikitext", "")
        # Truncate very long articles to avoid token limits
        if len(wikitext) > 100000:
            wikitext = wikitext[:100000] + "\n[... article truncated ...]"

        messages = build_messages(prompt_config, wikitext, topic)

        logger.info("[%d/%d] %s/%s — classifying with %d models...", i + 1, len(articles), drug, lang, len(models))

        results = []
        for model in models:
            result = classify_with_model(client, model, messages, drug, lang, topic)
            results.append(result)
            time.sleep(1)  # Rate limiting between models

        is_consensus, final_code, location = determine_consensus(results)

        # Write classification row
        cls_row = {
            "drug": drug, "lang": lang, "topic": topic,
            "model_1": models[0], "code_1": results[0]["code"],
            "model_2": models[1], "code_2": results[1]["code"],
            "model_3": models[2], "code_3": results[2]["code"],
            "consensus": "true" if is_consensus else "false", "final_code": final_code,
            "location": location, "timestamp": timestamp,
        }
        append_classification(cls_row)

        if is_consensus:
            n_consensus += 1
        else:
            n_disagreement += 1
            # Also write disagreement row for human review
            dis_row = {
                "drug": drug, "lang": lang, "topic": topic,
                "model_1": models[0], "code_1": results[0]["code"],
                "justification_1": results[0]["justification"],
                "model_2": models[1], "code_2": results[1]["code"],
                "justification_2": results[1]["justification"],
                "model_3": models[2], "code_3": results[2]["code"],
                "justification_3": results[2]["justification"],
                "human_code": "", "human_reviewer": "",
            }
            append_disagreement(dis_row)

        n_classified += 1

        codes_str = ", ".join(f"{m.split('/')[-1]}={r['code']}" for m, r in zip(models, results))
        consensus_str = "CONSENSUS" if is_consensus else "DISAGREEMENT"
        logger.info("  → %s: final_code=%d (%s)", consensus_str, final_code, codes_str)

    # Summary
    print(f"\n{'='*50}")
    print(f"Classification complete for topic: {topic}")
    print(f"  Total articles: {len(articles)}")
    print(f"  Classified: {n_classified}")
    print(f"  Skipped (already done): {n_skipped}")
    print(f"  Auto-coded 0 (missing): {n_auto_zero}")
    print(f"  Consensus: {n_consensus}")
    print(f"  Disagreement: {n_disagreement}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-based warning classification pipeline"
    )
    parser.add_argument(
        "--topic",
        required=True,
        choices=["g6pd", "teratogenicity", "qt"],
        help="Safety topic to classify",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-classify already-processed articles",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated OpenRouter model names (default: %(default)s)",
    )

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    if len(models) != 3:
        logger.error("Exactly 3 models required, got %d", len(models))
        sys.exit(1)

    run_classification(topic=args.topic, models=models, force=args.force)


if __name__ == "__main__":
    main()
