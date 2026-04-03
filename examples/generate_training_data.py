"""
Synthetic Training Data Generator for DataFrame QA
====================================================

Uses only dataframe metadata (column names, dtypes, descriptions) to generate
question/Pandas-query training pairs via the Claude API — no real data needed.

This replicates and extends the paper's UCI-DataFrameQA generation approach
(Section 4.1, Figure 6), where GPT-4 generated question/query pairs from
metadata for three user roles: Data Scientist, General User, Data Owner.

Why this matters:
  - Train smaller models to write Pandas queries WITHOUT access to real data
  - Generate domain-specific training sets from just a schema
  - Scale to any domain (medical, finance, automotive, etc.)
  - The generated pairs can fine-tune open-source models like Llama or Mistral

Usage:
    python generate_training_data.py
    python generate_training_data.py --data data/sample.csv --pairs-per-role 10
    python generate_training_data.py --schema-only --columns "age,sex,bp,cholesterol,target" --dtypes "int,str,float,float,int"
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from anthropic import Anthropic


client = Anthropic()
MODEL = "claude-sonnet-4-20250514"


# ============================================================================
# The three user roles from the paper (Table 4)
# ============================================================================

ROLES = {
    "data_scientist": {
        "name": "Data Scientist",
        "description": (
            "Questions tailored for individuals with an in-depth understanding "
            "of the dataset, possessing expertise in statistical and mathematical "
            "analysis. These questions should challenge their analytical skills, "
            "encouraging the use of advanced data manipulation and interpretation "
            "techniques. The focus is on extracting complex insights and patterns "
            "from the data."
        ),
    },
    "general_user": {
        "name": "General User",
        "description": (
            "Questions designed for users who may not have specialized data "
            "analysis skills but are interested in the practical, consumer-oriented "
            "aspects of the data. These questions should be somewhat open-ended, "
            "avoiding direct references to specific column names, thus introducing "
            "a level of interpretative ambiguity."
        ),
    },
    "data_owner": {
        "name": "Data Owner",
        "description": (
            "Questions aimed at individuals or entities who own or have created "
            "the data, with a focus on business-oriented insights. These questions "
            "should cater to their interest in understanding broader business "
            "implications, trends, and strategic insights that can be derived "
            "from the data."
        ),
    },
}


# ============================================================================
# Generate question/query pairs from metadata
# ============================================================================

def generate_pairs_for_role(
    columns,
    dtypes,
    role_key,
    num_pairs=10,
    dataset_name=None,
    dataset_description=None,
    column_descriptions=None,
):
    """
    Generate question/Pandas-query training pairs for a specific user role.

    This mirrors the paper's approach in Figure 6 — sending schema metadata
    and role descriptions to the LLM to produce realistic QA pairs.
    """
    role = ROLES[role_key]

    # Build the generation prompt (based on paper's Figure 6)
    prompt = f"""You are given a dataframe and are tasked with generating real-world \
questions and corresponding Pandas queries for the {role['name']} role.

The dataframe is described as follows:"""

    if dataset_name:
        prompt += f"\n\n- Name of dataframe: {dataset_name}"

    if dataset_description:
        prompt += f"\n- Description: {dataset_description}"

    prompt += f"\n\n- Column information:"
    for col, dt in zip(columns, dtypes):
        desc = ""
        if column_descriptions and col in column_descriptions:
            desc = f" — {column_descriptions[col]}"
        prompt += f"\n  {col} ({dt}){desc}"

    prompt += f"""

- Characteristics of the questions from the {role['name']} category:
{role['description']}

- Guidelines:
  - All questions must be solvable using the Pandas library in Python.
  - Questions should encompass a wide range of Pandas operations, from basic to advanced.
  - Questions must reflect real-world interests of the specified role.

- Assumptions:
  - The Pandas library has been imported as 'pd'. You can reference it directly.
  - The dataframe 'df' is loaded and available for use.
  - All string values in 'df' have been converted to lowercase.

- Response:
  - Store each answer in a variable named 'result'.
  - Do NOT include comments or explanations in the code.
  - Return ONLY valid JSON — no markdown fences, no extra text.

Generate exactly {num_pairs} question/query pairs. Return them as a JSON array:
[
  {{"question": "...", "query": "result = ..."}},
  ...
]"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.7,  # some creativity for diverse questions
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Clean markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)

    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  Warning: Failed to parse JSON for {role_key}. Raw response saved.")
        pairs = [{"raw_response": raw, "parse_error": True}]

    # Tag each pair with its role
    for pair in pairs:
        pair["role"] = role_key

    return pairs, message.usage


# ============================================================================
# Full generation pipeline
# ============================================================================

def generate_training_data(
    columns,
    dtypes,
    pairs_per_role=10,
    roles=None,
    dataset_name=None,
    dataset_description=None,
    column_descriptions=None,
):
    """
    Generate a complete training dataset from schema metadata.

    Returns question/query pairs for each role, plus token usage stats.
    """
    if roles is None:
        roles = list(ROLES.keys())

    all_pairs = []
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"\n{'='*60}")
    print(f"  Training Data Generation")
    print(f"  Model:          {MODEL}")
    print(f"  Columns:        {columns}")
    print(f"  Roles:          {roles}")
    print(f"  Pairs per role: {pairs_per_role}")
    print(f"  Total expected: {pairs_per_role * len(roles)}")
    print(f"{'='*60}\n")

    for role_key in roles:
        role_name = ROLES[role_key]["name"]
        print(f"Generating {pairs_per_role} pairs for {role_name}...")

        start = time.time()
        pairs, usage = generate_pairs_for_role(
            columns=columns,
            dtypes=dtypes,
            role_key=role_key,
            num_pairs=pairs_per_role,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            column_descriptions=column_descriptions,
        )
        elapsed = time.time() - start

        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        valid = [p for p in pairs if "parse_error" not in p]
        print(f"  Got {len(valid)} valid pairs ({elapsed:.1f}s, {usage.input_tokens + usage.output_tokens} tokens)\n")

        all_pairs.extend(pairs)

    # Classify question types (simple heuristic matching the paper's categories)
    for pair in all_pairs:
        if "parse_error" in pair:
            continue
        query = pair.get("query", "").lower()
        if any(op in query for op in ["groupby", "corr", "pivot", "merge", "apply", "rolling"]):
            pair["complexity"] = "data_analysis"
        elif any(op in query for op in [".sum()", ".count()", ".mean()", ".max()", ".min()"]):
            pair["complexity"] = "aggregation"
        else:
            pair["complexity"] = "retrieval"

    # Stats
    valid_pairs = [p for p in all_pairs if "parse_error" not in p]
    complexity_counts = {}
    for p in valid_pairs:
        c = p.get("complexity", "unknown")
        complexity_counts[c] = complexity_counts.get(c, 0) + 1

    print(f"{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"  Total pairs:    {len(valid_pairs)}")
    print(f"  Input tokens:   {total_input_tokens:,}")
    print(f"  Output tokens:  {total_output_tokens:,}")
    print(f"  Total tokens:   {total_input_tokens + total_output_tokens:,}")
    print(f"")
    print(f"  Complexity breakdown:")
    for c, count in sorted(complexity_counts.items()):
        pct = count / len(valid_pairs) * 100 if valid_pairs else 0
        print(f"    {c:20s} {count:>4} ({pct:.0f}%)")
    print(f"{'='*60}\n")

    return {
        "metadata": {
            "model": MODEL,
            "columns": columns,
            "dtypes": dtypes,
            "dataset_name": dataset_name,
            "dataset_description": dataset_description,
            "column_descriptions": column_descriptions,
            "pairs_per_role": pairs_per_role,
            "total_pairs": len(valid_pairs),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "complexity_breakdown": complexity_counts,
        },
        "pairs": all_pairs,
    }


# ============================================================================
# Validate generated pairs by executing them
# ============================================================================

def validate_pairs(pairs, df):
    """
    Execute each generated query against the actual dataframe to check validity.

    This is the quality control step — remove pairs that error out.
    """
    import numpy as np
    import math as math_mod

    print(f"Validating {len(pairs)} pairs against dataframe...")

    valid = []
    invalid = []

    working_df = df.copy()
    for col in working_df.select_dtypes(include=["object"]).columns:
        working_df[col] = working_df[col].str.lower()

    for pair in pairs:
        if "parse_error" in pair:
            invalid.append({**pair, "validation_error": "Failed to parse"})
            continue

        code = pair.get("query", "")
        safe_globals = {
            "pd": pd, "np": np, "math": math_mod,
            "df": working_df.copy(), "__builtins__": {},
        }
        local_vars = {}

        try:
            exec(code, safe_globals, local_vars)
            if "result" in local_vars:
                pair["validated"] = True
                pair["sample_result"] = str(local_vars["result"])[:200]
                valid.append(pair)
            else:
                pair["validation_error"] = "No 'result' variable"
                invalid.append(pair)
        except Exception as e:
            pair["validation_error"] = f"{type(e).__name__}: {e}"
            invalid.append(pair)

    pct = len(valid) / len(pairs) * 100 if pairs else 0
    print(f"  Valid:   {len(valid)}/{len(pairs)} ({pct:.0f}%)")
    print(f"  Invalid: {len(invalid)}/{len(pairs)}\n")

    if invalid:
        print(f"  Sample failures:")
        for inv in invalid[:3]:
            print(f"    Q: {inv.get('question', 'N/A')}")
            print(f"    Error: {inv.get('validation_error', 'N/A')}\n")

    return valid, invalid


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from dataframe metadata"
    )
    parser.add_argument("--data", default="data/sample.csv", help="Path to CSV")
    parser.add_argument("--pairs-per-role", type=int, default=5, help="Questions per role")
    parser.add_argument("--roles", nargs="+", default=None,
                        choices=["data_scientist", "general_user", "data_owner"])
    parser.add_argument("--dataset-name", default=None, help="Name of the dataset")
    parser.add_argument("--dataset-description", default=None, help="Description of the dataset")
    parser.add_argument("--validate", action="store_true", help="Validate pairs against data")
    parser.add_argument("--output", default="results/training_data.json", help="Output path")

    # Schema-only mode (no CSV needed)
    parser.add_argument("--schema-only", action="store_true",
                        help="Provide schema manually instead of loading a CSV")
    parser.add_argument("--columns", default=None,
                        help="Comma-separated column names (for --schema-only)")
    parser.add_argument("--dtypes", default=None,
                        help="Comma-separated dtypes (for --schema-only)")

    args = parser.parse_args()

    # Get schema
    if args.schema_only:
        if not args.columns or not args.dtypes:
            parser.error("--schema-only requires --columns and --dtypes")
        columns = [c.strip() for c in args.columns.split(",")]
        dtypes = [d.strip() for d in args.dtypes.split(",")]
        df = None
        print(f"Schema-only mode: {len(columns)} columns")
    else:
        df = pd.read_csv(args.data)
        columns = df.columns.tolist()
        dtypes = [str(dt) for dt in df.dtypes.tolist()]
        print(f"Loaded {len(df)} rows x {len(columns)} columns from {args.data}")

    # Generate
    output = generate_training_data(
        columns=columns,
        dtypes=dtypes,
        pairs_per_role=args.pairs_per_role,
        roles=args.roles,
        dataset_name=args.dataset_name or args.data,
        dataset_description=args.dataset_description,
    )

    # Optionally validate against real data
    if args.validate and df is not None:
        valid, invalid = validate_pairs(output["pairs"], df)
        output["metadata"]["validated"] = True
        output["metadata"]["valid_pairs"] = len(valid)
        output["metadata"]["invalid_pairs"] = len(invalid)
        output["pairs"] = valid  # keep only validated pairs

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved {len(output['pairs'])} pairs to {args.output}")

    # Show some examples
    print(f"\nSample generated pairs:")
    for pair in output["pairs"][:3]:
        if "parse_error" in pair:
            continue
        print(f"  [{pair.get('role', '?')}] [{pair.get('complexity', '?')}]")
        print(f"  Q: {pair['question']}")
        print(f"  A: {pair['query']}\n")
