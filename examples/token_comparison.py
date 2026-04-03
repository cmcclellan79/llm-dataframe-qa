"""
Token Usage Comparison: DataFrame QA vs Traditional Full-Table Approach
========================================================================

Runs the same questions two ways:
  1. DataFrame QA (metadata only) — sends column names + dtypes
  2. Traditional (full table)     — embeds the entire table in the prompt

Compares token usage, cost, and accuracy side by side.

Usage:
    python token_comparison.py
    python token_comparison.py --data data/sample.csv --questions data/sample_qa.json
"""

import argparse
import json
import re
import time

import pandas as pd
import numpy as np
import math
from anthropic import Anthropic


client = Anthropic()
MODEL = "claude-sonnet-4-20250514"


# ============================================================================
# Shared: system prompt, code cleaning, sandbox, comparison
# ============================================================================

SYSTEM_PROMPT = """You are a professional Python programming assistant. \
Write Pandas code to get the answer to the user's question.
- Assumptions:
  - The Pandas library has been imported as 'pd'. You can reference it directly.
  - The dataframe 'df' is loaded and available for use.
  - All string values in the 'df' have been converted to lowercase.
- Requirements:
  - Use only Pandas operations for the solution.
  - Store the answer in a variable named 'result'.
  - Do NOT include comments or explanations in your code.
  - Do NOT import any libraries.
  - Return ONLY the Python code, nothing else."""


def clean_code(raw):
    code = raw.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)
    code = re.sub(r"^\[PYTHON\]\s*\n?", "", code)
    code = re.sub(r"\n?\[/PYTHON\]\s*$", "", code)
    return code.strip()


def execute_query(code, df):
    safe_globals = {
        "pd": pd, "np": np, "math": math,
        "df": df.copy(), "__builtins__": {},
    }
    local_vars = {}
    try:
        exec(code, safe_globals, local_vars)
        result = local_vars.get("result")
        if result is None and "result" not in local_vars:
            return {"success": False, "result": None, "error": "No 'result' variable"}
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": f"{type(e).__name__}: {e}"}


def compare_results(predicted, expected):
    if isinstance(predicted, (pd.Series, pd.DataFrame)):
        values = predicted.values.flatten().tolist()
        str_values = [str(v).lower().strip() for v in values]
        str_expected = str(expected).lower().strip()
        if str_expected in str_values:
            return True
        try:
            float_exp = float(expected)
            for v in values:
                try:
                    if abs(float(v) - float_exp) < 1e-6:
                        return True
                except (ValueError, TypeError):
                    continue
        except (ValueError, TypeError):
            pass
        return False
    if isinstance(predicted, str) or isinstance(expected, str):
        return str(predicted).lower().strip() == str(expected).lower().strip()
    try:
        return abs(float(predicted) - float(expected)) < 1e-6
    except (ValueError, TypeError):
        return str(predicted) == str(expected)


# ============================================================================
# Approach 1: DataFrame QA (metadata only)
# ============================================================================

def run_metadata_only(df, question):
    """Paper's approach: send only column names and dtypes."""
    columns = df.columns.tolist()
    dtypes = [str(dt) for dt in df.dtypes.tolist()]

    user_prompt = f"""You are given a Pandas dataframe named 'df':
- Columns: {columns}
- Data Types: {dtypes}
- User's Question: {question}"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    code = clean_code(message.content[0].text)
    result = execute_query(code, df)

    return {
        "code": code,
        "result": result,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
    }


# ============================================================================
# Approach 2: Traditional (full table embedded in prompt)
# ============================================================================

def run_full_table(df, question):
    """Traditional approach: embed the entire table as CSV in the prompt."""
    table_csv = df.to_csv(index=False)

    user_prompt = f"""You are given a Pandas dataframe named 'df'.

Here is the full table data in CSV format:
{table_csv}

- User's Question: {question}"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    code = clean_code(message.content[0].text)
    result = execute_query(code, df)

    return {
        "code": code,
        "result": result,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
    }


# ============================================================================
# Run comparison
# ============================================================================

def run_comparison(df, qa_pairs):
    # Lowercase strings
    working_df = df.copy()
    for col in working_df.select_dtypes(include=["object"]).columns:
        working_df[col] = working_df[col].str.lower()

    total_meta_input = 0
    total_meta_output = 0
    total_full_input = 0
    total_full_output = 0
    meta_correct = 0
    full_correct = 0

    print(f"\n{'='*72}")
    print(f"  TOKEN COMPARISON: Metadata-Only vs Full-Table")
    print(f"  Model: {MODEL}")
    print(f"  Questions: {len(qa_pairs)} | Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"{'='*72}\n")

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        expected = qa["expected_answer"]

        print(f"[{i+1}/{len(qa_pairs)}] {question}")

        # --- Metadata-only approach ---
        meta = run_metadata_only(working_df, question)
        meta_pass = meta["result"]["success"] and compare_results(meta["result"]["result"], expected)
        if meta_pass:
            meta_correct += 1

        total_meta_input += meta["input_tokens"]
        total_meta_output += meta["output_tokens"]

        # --- Full-table approach ---
        full = run_full_table(working_df, question)
        full_pass = full["result"]["success"] and compare_results(full["result"]["result"], expected)
        if full_pass:
            full_correct += 1

        total_full_input += full["input_tokens"]
        total_full_output += full["output_tokens"]

        # Per-question comparison
        savings = full["input_tokens"] - meta["input_tokens"]
        pct = (savings / full["input_tokens"] * 100) if full["input_tokens"] > 0 else 0

        print(f"  Metadata:   {meta['total_tokens']:>5} tokens (in:{meta['input_tokens']} + out:{meta['output_tokens']})  {'PASS' if meta_pass else 'FAIL'}")
        print(f"  Full table: {full['total_tokens']:>5} tokens (in:{full['input_tokens']} + out:{full['output_tokens']})  {'PASS' if full_pass else 'FAIL'}")
        print(f"  Saved:      {savings} input tokens ({pct:.0f}% reduction)\n")

    # Summary
    total_meta = total_meta_input + total_meta_output
    total_full = total_full_input + total_full_output
    total_saved = total_full - total_meta
    pct_saved = (total_saved / total_full * 100) if total_full > 0 else 0

    print(f"{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"")
    print(f"  {'':30s} {'Metadata-Only':>15s} {'Full-Table':>15s}")
    print(f"  {'─'*62}")
    print(f"  {'Total input tokens':30s} {total_meta_input:>15,} {total_full_input:>15,}")
    print(f"  {'Total output tokens':30s} {total_meta_output:>15,} {total_full_output:>15,}")
    print(f"  {'Total tokens':30s} {total_meta:>15,} {total_full:>15,}")
    print(f"  {'Avg tokens per question':30s} {total_meta // len(qa_pairs):>15,} {total_full // len(qa_pairs):>15,}")
    print(f"  {'pass@1':30s} {meta_correct}/{len(qa_pairs):>13} {full_correct}/{len(qa_pairs):>13}")
    print(f"  {'─'*62}")
    print(f"  {'Token savings':30s} {total_saved:>15,} ({pct_saved:.0f}% reduction)")
    print(f"")

    # Privacy note
    print(f"  PRIVACY")
    print(f"  Metadata-only: Data NEVER sent to API  (only column names + types)")
    print(f"  Full-table:    ALL data sent to API     (privacy risk)")
    print(f"{'='*72}\n")

    # Save results
    results = {
        "model": MODEL,
        "num_questions": len(qa_pairs),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "metadata_only": {
            "input_tokens": total_meta_input,
            "output_tokens": total_meta_output,
            "total_tokens": total_meta,
            "pass_at_1": meta_correct / len(qa_pairs),
            "correct": meta_correct,
        },
        "full_table": {
            "input_tokens": total_full_input,
            "output_tokens": total_full_output,
            "total_tokens": total_full,
            "pass_at_1": full_correct / len(qa_pairs),
            "correct": full_correct,
        },
        "token_savings": total_saved,
        "token_savings_pct": round(pct_saved, 1),
    }

    with open("results/token_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to results/token_comparison.json")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare token usage: metadata-only vs full-table")
    parser.add_argument("--data", default="data/sample.csv", help="Path to CSV")
    parser.add_argument("--questions", default="data/sample_qa.json", help="Path to QA pairs JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    with open(args.questions) as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    print(f"Loaded {len(qa_pairs)} questions")

    run_comparison(df, qa_pairs)
