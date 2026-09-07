#!/usr/bin/env bash
# Run all 24 LLM queries for a given model and capture screenshots.
#
# Usage:
#   bash scripts/run_llm_batch.sh --model {chatgpt,claude,gemini} [--wait 30]
#
# Requires: .env with credentials, conda env 'pharma' active.

set -euo pipefail

MODEL=""
WAIT=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --wait)  WAIT="$2";  shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 --model {chatgpt,claude,gemini} [--wait 30]" >&2
    exit 1
fi

DRUGS=(nitrofurantoin primaquine dapsone)
LANGS=(en he ar ru)
PROMPT_TYPES=(A B)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

PASS=0
FAIL=0

for drug in "${DRUGS[@]}"; do
    for lang in "${LANGS[@]}"; do
        for pt in "${PROMPT_TYPES[@]}"; do
            echo "--- ${MODEL} | ${lang} | ${drug} | ${pt} ---"
            if python "${SCRIPT_DIR}/llm_browser_session.py" \
                --model "$MODEL" \
                --lang "$lang" \
                --drug "$drug" \
                --prompt-type "$pt" \
                --wait "$WAIT"; then
                PASS=$((PASS + 1))
            else
                echo "FAILED: ${MODEL} ${lang} ${drug} ${pt}" >&2
                FAIL=$((FAIL + 1))
            fi
        done
    done
done

echo ""
echo "Done: ${PASS} succeeded, ${FAIL} failed."
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
