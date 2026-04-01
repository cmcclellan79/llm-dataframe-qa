"""
DataFrame QA — Full API Walkthrough
====================================

A complete, runnable script that replicates the DataFrame QA paper's approach
using the Anthropic Claude API.

Paper: Ye, Du & Wang (2024)
"DataFrame QA: A Universal LLM Framework on DataFrame Question Answering
Without Data Exposure" — https://arxiv.org/abs/2401.15463

Setup:
    pip install anthropic pandas numpy tqdm
    export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"

Usage:
    python api_walkthrough.py
"""

import os
import json
import time
import re

import pandas as pd
import numpy as np
import math
from anthropic import Anthropic


# STEP 1: Initialize the client


client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

MODEL = "claude-sonnet-4-20250514"  # cost-effective choice
# MODEL = "claude-opus-4-6"         # highest accuracy (uncomment to use)


# STEP 2: Build the prompt (metadata only — no data leaves your machine)


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


def build_user_prompt(columns, dtypes, question, column_descriptions=None):
    """
    Build the user prompt from schema metadata only.
    
    The key insight from the paper: we send ONLY column names and data types
    to the LLM. The actual data never leaves your machine.
    """
    prompt = f"""You are given a Pandas dataframe named 'df':
- Columns: {columns}
- Data Types: {dtypes}"""

    # Optional: add column descriptions to reduce ambiguity errors
    if column_descriptions:
        for col, desc in column_descriptions.items():
            prompt += f"\n- Column '{col}': {desc}"

    prompt += f"\n- User's Question: {question}"
    return prompt



# STEP 3: Call the Claude API


def generate_pandas_query(columns, dtypes, question, column_descriptions=None):
    """
    Send metadata to Claude and get back executable Pandas code.
    
    Uses temperature=0 for greedy decoding (deterministic output),
    matching the paper's methodology for pass@1 evaluation.
    """
    user_prompt = build_user_prompt(columns, dtypes, question, column_descriptions)

    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()

    # Clean markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:python)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

    # Clean [PYTHON] tags (paper format)
    raw = re.sub(r"^\[PYTHON\]\s*\n?", "", raw)
    raw = re.sub(r"\n?\[/PYTHON\]\s*$", "", raw)

    return raw.strip()



# STEP 4: Execute in a sandbox (restricted environment)


def execute_query(code, df):
    """
    Execute generated Pandas code in a restricted namespace.
    
    Only pandas, numpy, and math are available — matching the paper's
    sandbox constraints. This prevents arbitrary code execution.
    """
    safe_globals = {
        "pd": pd,
        "np": np,
        "math": math,
        "df": df.copy(),  # work on a copy to prevent mutation
        "__builtins__": {},
    }
    local_vars = {}

    try:
        exec(code, safe_globals, local_vars)
        result = local_vars.get("result")
        if result is None and "result" not in local_vars:
            return {"success": False, "result": None, "error": "No 'result' variable found"}
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": f"{type(e).__name__}: {e}"}



# STEP 5: Evaluate results (pass@1)


def compare_results(predicted, expected):
    """
    Relaxed comparison matching the paper's evaluation criteria.
    
    Handles:
      - Series/DataFrame containment (paper considers these correct if they
        include the answer)
      - Case-insensitive string matching
      - Numeric comparison with tolerance
    """
    # Handle Series/DataFrame — check if expected is contained in result
    if isinstance(predicted, (pd.Series, pd.DataFrame)):
        values = predicted.values.flatten().tolist()
        str_values = [str(v).lower().strip() for v in values]
        str_expected = str(expected).lower().strip()
        if str_expected in str_values:
            return True
        # Try numeric containment
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

    # String comparison (case-insensitive)
    if isinstance(predicted, str) or isinstance(expected, str):
        return str(predicted).lower().strip() == str(expected).lower().strip()

    # Numeric comparison (with tolerance)
    try:
        return abs(float(predicted) - float(expected)) < 1e-6
    except (ValueError, TypeError):
        return str(predicted) == str(expected)



# STEP 6: Full evaluation loop


def run_evaluation(df, qa_pairs):
    """
    Run the complete DataFrame QA pipeline on a set of question/answer pairs.
    
    For each question:
      1. Extract metadata from the dataframe
      2. Send to Claude API (metadata only)
      3. Execute the returned code in the sandbox
      4. Compare result to ground truth
      5. Compute pass@1
    """
    # Lowercase all strings (paper preprocessing step)
    working_df = df.copy()
    for col in working_df.select_dtypes(include=["object"]).columns:
        working_df[col] = working_df[col].str.lower()

    columns = working_df.columns.tolist()
    dtypes = [str(dt) for dt in working_df.dtypes.tolist()]

    correct = 0
    total = len(qa_pairs)
    errors = []

    print(f"\n{'='*60}")
    print(f"  DataFrame QA Evaluation")
    print(f"  Model:     {MODEL}")
    print(f"  Questions: {total}")
    print(f"  Columns:   {columns}")
    print(f"{'='*60}\n")

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        expected = qa["expected_answer"]

        print(f"[{i+1}/{total}] {question}")

        # Generate query via API
        start = time.time()
        code = generate_pandas_query(columns, dtypes, question)
        api_time = time.time() - start

        print(f"  Code: {code}")

        # Execute in sandbox
        output = execute_query(code, working_df)

        if output["success"]:
            is_correct = compare_results(output["result"], expected)
            status = "PASS" if is_correct else "FAIL"

            if is_correct:
                correct += 1
            else:
                errors.append({
                    "question": question,
                    "code": code,
                    "got": str(output["result"]),
                    "expected": str(expected),
                })

            print(f"  Result: {output['result']}")
            print(f"  Expected: {expected}")
            print(f"  [{status}] ({api_time:.1f}s)\n")
        else:
            errors.append({
                "question": question,
                "code": code,
                "error": output["error"],
                "expected": str(expected),
            })
            print(f"  ERROR: {output['error']}")
            print(f"  [FAIL] ({api_time:.1f}s)\n")

    # Final score
    pass_at_1 = correct / total if total > 0 else 0

    print(f"{'='*60}")
    print(f"  RESULTS")
    print(f"  pass@1: {pass_at_1:.1%} ({correct}/{total})")
    print(f"{'='*60}")

    if errors:
        print(f"\n  Failures ({len(errors)}):")
        for e in errors:
            print(f"    - {e['question']}")
            if "error" in e:
                print(f"      Error: {e['error']}")
            else:
                print(f"      Got: {e['got']} | Expected: {e['expected']}")

    # Save results
    results = {
        "model": MODEL,
        "pass_at_1": pass_at_1,
        "correct": correct,
        "total": total,
        "errors": errors,
    }
    with open("results/walkthrough_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to results/walkthrough_results.json")

    return pass_at_1



# MAIN — Run the demo


if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv("data/sample.csv")

    # Load sample QA pairs
    with open("data/sample_qa.json") as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    print(f"Loaded {len(qa_pairs)} question/answer pairs")

    # Run the full pipeline
    run_evaluation(df, qa_pairs)
