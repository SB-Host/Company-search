#!/usr/bin/env python3
"""
Fill missing city, state, public/private, and description for companies.
Uses Claude web search. Run from the company-search folder.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python3 fill_locations.py
    ANTHROPIC_API_KEY=sk-ant-... python3 fill_locations.py --sample 10
"""

import argparse
import json
import os
import re
import sys
import time

import anthropic
import pandas as pd

CSV_FILE  = "/Users/sawyerbrooks/company-search/cleaned_companies.csv"
MODEL     = "claude-sonnet-4-6"
DELAY     = 2   # seconds between API calls (keeps under rate limit)

PROMPT_TEMPLATE = """Find the headquarters city, state, whether the company is public or private, and a short description for this company:

Company: {company}
Website: {website}

Search the web and return ONLY this JSON — nothing else:
{{"city": "<city or null>", "state": "<2-letter US state or null>", "public_private": "<Public, Private, Acquired, or null>", "description": "<under 20 words describing what they do, or null>"}}

Rules:
- state: 2-letter US abbreviation (CA, TX, NY etc). Null if not a US company.
- public_private: Public = listed on stock exchange, Private = privately held, Acquired = bought by another company
- description: under 20 words, factual, what industry and what they do
- Return the JSON object only, no explanation"""


def ask_claude(ai, company, website):
    prompt = PROMPT_TEMPLATE.format(company=company, website=website or "unknown")
    for attempt in range(4):
        try:
            response = ai.messages.create(
                model=MODEL,
                max_tokens=300,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": prompt}],
            )
            # Get the last text block
            for block in reversed(response.content):
                if hasattr(block, "text"):
                    match = re.search(r'\{[^{}]+\}', block.text, re.DOTALL)
                    if match:
                        return json.loads(match.group())
            return {}
        except anthropic.RateLimitError:
            wait = 30 + (attempt * 15)
            print(f"  [rate limit] waiting {wait}s...", flush=True)
            time.sleep(wait)
        except anthropic.AuthenticationError:
            print("ERROR: Invalid API key.", flush=True)
            sys.exit(1)
        except Exception as e:
            msg = str(e)
            if "credit balance is too low" in msg or "insufficient_quota" in msg:
                print("\nERROR: Out of API credits. Add credits at console.anthropic.com → Plans & Billing.", flush=True)
                sys.exit(1)
            print(f"  [error] {e}", flush=True)
            return {}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", flush=True)
        sys.exit(1)

    print(f"Loading {CSV_FILE}...", flush=True)
    df = pd.read_csv(CSV_FILE, dtype=str).fillna("")

    for col in ["city_clean", "state_clean", "public_private", "description"]:
        if col not in df.columns:
            df[col] = ""

    print(f"Loaded {len(df)} rows.", flush=True)

    mask = (
        (df["city_clean"] == "") |
        (df["state_clean"] == "") |
        (df["description"] == "")
    )
    targets = df[mask].copy()
    print(f"Rows needing data: {len(targets)}", flush=True)

    if args.sample:
        targets = targets.head(args.sample)
        print(f"Sample mode: {args.sample} rows", flush=True)

    if targets.empty:
        print("Nothing to do.", flush=True)
        return

    ai      = anthropic.Anthropic(api_key=api_key)
    total   = len(targets)
    filled  = 0

    print(f"\nStarting... ({DELAY}s between requests)\n" + "-"*60, flush=True)

    for i, (idx, row) in enumerate(targets.iterrows(), 1):
        company = str(row.get("Company", "")).strip()
        website = str(row.get("Website", "")).strip()

        print(f"[{i}/{total}] {company[:55]}", end="  ", flush=True)

        result = ask_claude(ai, company, website)

        new_city    = result.get("city")           or ""
        new_state   = result.get("state")          or ""
        new_pp      = result.get("public_private") or ""
        new_desc    = result.get("description")    or ""

        # Only overwrite if we got something new
        final_city  = new_city  or str(row.get("city_clean",    "")).strip()
        final_state = new_state or str(row.get("state_clean",   "")).strip()
        final_pp    = new_pp    or str(row.get("public_private","")).strip()
        final_desc  = new_desc  or str(row.get("description",   "")).strip()

        df.at[idx, "city_clean"]    = final_city
        df.at[idx, "state_clean"]   = final_state
        df.at[idx, "public_private"] = final_pp
        df.at[idx, "description"]   = final_desc

        if new_city or new_state or new_pp or new_desc:
            filled += 1
            print(f"{final_city}, {final_state}  |  {final_pp}", flush=True)
        else:
            print("NOT FOUND", flush=True)

        # Save every 10 rows
        if i % 10 == 0:
            df.to_csv(CSV_FILE, index=False)
            print(f"  -- saved checkpoint {i}/{total} --", flush=True)

        time.sleep(DELAY)

    df.to_csv(CSV_FILE, index=False)

    print(f"\n{'─'*60}", flush=True)
    print(f"Done. Filled {filled}/{total} rows.", flush=True)
    still = ((df["city_clean"] == "") | (df["state_clean"] == "")).sum()
    print(f"Still missing city/state: {still}", flush=True)
    print(f"Saved to {CSV_FILE}", flush=True)


if __name__ == "__main__":
    main()
