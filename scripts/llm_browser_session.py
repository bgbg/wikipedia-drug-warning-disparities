#!/usr/bin/env python3
"""
Submit a single LLM query via a web interface and capture a screenshot.

All three apps (ChatGPT, Claude, Gemini) are accessed via Google SSO.
Sessions are persisted to avoid repeated 2FA prompts.

FIRST-TIME SETUP (run once per app):
    python scripts/llm_browser_session.py --model claude --setup-session
    python scripts/llm_browser_session.py --model chatgpt --setup-session
    python scripts/llm_browser_session.py --model gemini --setup-session

    Each opens a browser window. Sign in with Google and complete 2FA manually.
    Press Enter in the terminal when the app home page is fully loaded.
    The session is saved to data/sessions/{model}_session.json.

NORMAL USAGE (after setup):
    python scripts/llm_browser_session.py \
        --model {chatgpt,claude,gemini} \
        --lang {en,he,ar,ru} \
        --drug {nitrofurantoin,primaquine,dapsone} \
        --prompt-type {A,B} \
        [--wait 30]

Credentials read from .env: GOOGLE_EMAIL, GOOGLE_PASSWORD
Sessions stored in: data/sessions/{model}_session.json  (gitignored)
Screenshots saved to: data/llm_screenshots/{model}_{lang}_{drug}_{prompt_type}_{date}.png
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

load_dotenv()

ROOT = Path(__file__).parent.parent
PROMPTS_FILE = ROOT / "data" / "prompts" / "llm_prompts.json"
SCREENSHOTS_DIR = ROOT / "data" / "llm_screenshots"
SESSIONS_DIR = ROOT / "data" / "sessions"

MODEL_URLS = {
    "chatgpt": "https://chatgpt.com/",
    "claude":  "https://claude.ai/new",
    "gemini":  "https://gemini.google.com/app",
}

MODEL_LOGIN_URLS = {
    "chatgpt": "https://chatgpt.com/auth/login",
    "claude":  "https://claude.ai/login",
    "gemini":  "https://accounts.google.com/signin",
}

MAX_RETRIES = 3
RETRY_DELAY = 10


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def session_file(model: str) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"{model}_session.json"


def setup_session(model: str):
    """
    Open a browser, navigate to the app login page, and wait for the user
    to complete sign-in (including Google 2FA) manually. Then save the session.
    """
    print(f"\nSetting up session for {model}.")
    print("A browser window will open. Sign in with Google and complete 2FA.")
    print("When the app home page is fully loaded, come back here and press Enter.\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        context = browser.new_context()
        page = context.new_page()

        page.goto(MODEL_LOGIN_URLS[model], wait_until="networkidle")

        input("  >>> Browser is open. Complete sign-in and 2FA, then press Enter here: ")

        out = session_file(model)
        context.storage_state(path=str(out))
        browser.close()

    print(f"Session saved to {out}")
    print("You can now run queries without being asked for 2FA again (until the session expires).\n")


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def load_prompt(drug: str, lang: str, prompt_type: str) -> str:
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    return prompts[drug][lang][prompt_type]


def screenshot_path(model: str, lang: str, drug: str, prompt_type: str) -> Path:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    return SCREENSHOTS_DIR / f"{model}_{lang}_{drug}_{prompt_type}_{today}.png"


def submit_and_screenshot(page, model: str, prompt: str, wait: int, out_path: Path) -> str:
    """Navigate to new chat, submit prompt, wait, screenshot. Returns detected model version."""
    page.goto(MODEL_URLS[model], wait_until="networkidle")
    page.wait_for_timeout(2000)

    selectors = [
        "div[contenteditable='true']",
        "textarea[placeholder]",
        "textarea",
    ]
    input_el = None
    for sel in selectors:
        try:
            input_el = page.wait_for_selector(sel, timeout=8000)
            break
        except PlaywrightTimeoutError:
            continue

    if input_el is None:
        raise RuntimeError("Could not locate prompt input field")

    input_el.click()
    input_el.fill(prompt)
    page.keyboard.press("Enter")
    time.sleep(wait)

    page.screenshot(path=str(out_path), full_page=False)

    version = "unknown"
    for sel in ["[data-testid='model-switcher']", ".model-name", "title"]:
        try:
            el = page.query_selector(sel)
            if el:
                version = (el.inner_text() or el.get_attribute("content") or "unknown").strip()
                break
        except Exception:
            pass

    return version


def run_query(model: str, lang: str, drug: str, prompt_type: str, wait: int):
    sf = session_file(model)
    if not sf.exists():
        print(f"ERROR: No saved session for {model}.", file=sys.stderr)
        print(f"Run first:  python {__file__} --model {model} --setup-session", file=sys.stderr)
        sys.exit(1)

    prompt = load_prompt(drug, lang, prompt_type)
    out_path = screenshot_path(model, lang, drug, prompt_type)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False, slow_mo=50)
                context = browser.new_context(
                    storage_state=str(sf),
                    locale={"he": "he-IL", "ar": "ar-SA"}.get(lang, lang),
                )
                page = context.new_page()
                version = submit_and_screenshot(page, model, prompt, wait, out_path)
                browser.close()

            print(f"screenshot: {out_path}")
            print(f"model_version: {version}")
            return

        except Exception as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print("ERROR: all retries exhausted. Session may have expired — re-run --setup-session.", file=sys.stderr)
                sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Submit one LLM query via web interface and capture screenshot")
    parser.add_argument("--model", required=True, choices=["chatgpt", "claude", "gemini"])
    parser.add_argument("--setup-session", action="store_true",
                        help="Open browser for manual login + 2FA, then save session. Run once per app before querying.")
    parser.add_argument("--lang", choices=["en", "he", "ar", "ru"])
    parser.add_argument("--drug", choices=["nitrofurantoin", "primaquine", "dapsone"])
    parser.add_argument("--prompt-type", choices=["A", "B"], dest="prompt_type")
    parser.add_argument("--wait", type=int, default=30, help="Seconds to wait for LLM response (default: 30)")
    args = parser.parse_args()

    if args.setup_session:
        setup_session(args.model)
    else:
        for required in ("lang", "drug", "prompt_type"):
            if getattr(args, required) is None:
                parser.error(f"--{required.replace('_', '-')} is required when not using --setup-session")
        run_query(args.model, args.lang, args.drug, args.prompt_type, args.wait)


if __name__ == "__main__":
    main()
