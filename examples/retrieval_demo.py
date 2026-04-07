"""
DataFrame QA — Retrieval-Based Predictive Demo (Heart Disease)
===============================================================

Demonstrates the full pipeline:
  1. Generate training pairs from schema metadata (no real data)
  2. Build a TF-IDF retrieval index over generated question/query pairs
  3. For new questions, retrieve the most similar training example
  4. Use the retrieved query as a template — adapt it via a lightweight
     Claude call that only sees the template + new question (still no data)
  5. Execute the adapted query in the sandbox and return results

This shows how metadata-generated training data creates a reusable
knowledge base that reduces API dependency and keeps data private.

Usage:
    python examples/retrieval_demo.py

    # Interactive mode
    python examples/retrieval_demo.py --interactive

    # Skip generation (use cached training data)
    python examples/retrieval_demo.py --skip-generation
"""

import argparse
import json
import os
import re
import time
import sys

import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic


# ============================================================================
# Configuration
# ============================================================================

client = Anthropic()
MODEL = "claude-sonnet-4-20250514"
TRAINING_DATA_PATH = "results/heart_disease_training.json"

# Heart disease schema (metadata only — this is ALL the LLM ever sees)
SCHEMA = {
    "name": "Heart Disease Dataset",
    "columns": [
        "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
        "fasting_blood_sugar", "rest_ecg", "max_heart_rate",
        "exercise_angina", "oldpeak", "slope", "num_major_vessels",
        "thalassemia", "target"
    ],
    "dtypes": [
        "int", "str", "str", "float", "float",
        "bool", "str", "float",
        "bool", "float", "str", "int",
        "str", "int"
    ],
    "column_descriptions": {
        "target": "0 = no heart disease, 1-4 = increasing severity",
        "chest_pain_type": "typical angina, atypical angina, non-anginal pain, asymptomatic",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "num_major_vessels": "number of major vessels (0-3) colored by fluoroscopy",
        "rest_ecg": "resting electrocardiographic results",
        "thalassemia": "normal, fixed defect, or reversible defect",
        "slope": "slope of peak exercise ST segment",
    },
}

# Sample heart disease data for execution (stays local, never sent to API)
SAMPLE_DATA = {
    "age": [63, 37, 41, 56, 57, 67, 45, 68, 57, 38, 62, 53, 52, 44, 55, 48, 54, 65, 43, 50],
    "sex": ["male", "male", "female", "male", "female", "male", "female", "male", "male", "female",
            "female", "male", "male", "female", "male", "male", "female", "male", "female", "male"],
    "chest_pain_type": ["typical angina", "non-anginal pain", "atypical angina", "atypical angina",
                        "asymptomatic", "asymptomatic", "atypical angina", "non-anginal pain",
                        "asymptomatic", "typical angina", "asymptomatic", "asymptomatic",
                        "typical angina", "non-anginal pain", "asymptomatic", "atypical angina",
                        "non-anginal pain", "asymptomatic", "non-anginal pain", "asymptomatic"],
    "resting_bp": [145, 130, 130, 120, 120, 160, 112, 120, 140, 138, 140, 140, 172, 118, 132, 130, 135, 120, 132, 144],
    "cholesterol": [233, 250, 204, 236, 354, 286, 160, 229, 241, 175, 268, 203, 199, 242, 353, 245, 304, 177, 341, 200],
    "fasting_blood_sugar": [True, False, False, False, False, True, False, True, False, False,
                            False, True, True, False, False, False, True, False, True, False],
    "rest_ecg": ["normal", "normal", "normal", "normal", "normal", "normal", "normal", "normal",
                 "hypertrophy", "normal", "normal", "normal", "normal", "normal", "normal",
                 "normal", "normal", "normal", "normal", "normal"],
    "max_heart_rate": [150, 187, 172, 178, 131, 108, 138, 129, 123, 173, 160, 155, 162, 149, 132, 180, 170, 140, 136, 126],
    "exercise_angina": [False, False, False, False, True, True, False, True, True, False,
                        False, True, False, False, True, False, False, False, True, True],
    "oldpeak": [2.3, 3.5, 1.4, 0.8, 1.6, 1.5, 0.0, 2.6, 0.2, 0.0,
                3.6, 3.1, 0.5, 0.3, 1.2, 0.0, 0.0, 0.4, 3.0, 0.9],
    "slope": ["downsloping", "downsloping", "upsloping", "upsloping", "upsloping",
              "flat", "flat", "flat", "flat", "upsloping",
              "downsloping", "downsloping", "upsloping", "flat", "flat",
              "upsloping", "upsloping", "upsloping", "flat", "flat"],
    "num_major_vessels": [0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0],
    "thalassemia": ["fixed defect", "normal", "normal", "normal", "normal",
                    "normal", "normal", "reversible defect", "reversible defect", "normal",
                    "normal", "reversible defect", "reversible defect", "normal", "reversible defect",
                    "normal", "normal", "reversible defect", "reversible defect", "reversible defect"],
    "target": [1, 1, 0, 0, 2, 3, 0, 1, 2, 0, 3, 2, 0, 0, 2, 0, 0, 1, 2, 3],
}


# ============================================================================
# Step 1: Generate training pairs from metadata
# ============================================================================

ROLE_PROMPTS = {
    "data_scientist": "Expert analytical questions requiring groupby, correlation, pivot tables, statistical analysis, and multi-step transformations.",
    "general_user": "Practical questions a non-technical person (e.g. a patient) would ask, avoiding column names, focused on health insights.",
    "data_owner": "Business/clinical questions a hospital administrator would ask — trends, risk factors, resource allocation.",
}


def generate_training_pairs(num_per_role=10):
    """Generate question/query pairs from schema metadata only."""
    all_pairs = []
    col_info = "\n".join(
        f"  {col} ({dt})" + (f" — {SCHEMA['column_descriptions'].get(col, '')}" if SCHEMA['column_descriptions'].get(col) else "")
        for col, dt in zip(SCHEMA["columns"], SCHEMA["dtypes"])
    )

    print(f"\n  Generating training data from metadata...")
    print(f"  Columns: {len(SCHEMA['columns'])}")
    print(f"  Roles: {list(ROLE_PROMPTS.keys())}")
    print(f"  Pairs per role: {num_per_role}\n")

    for role, desc in ROLE_PROMPTS.items():
        prompt = f"""Generate {num_per_role} realistic question/Pandas-query pairs for the {role} role.

Dataset: {SCHEMA['name']}
Columns:
{col_info}

Role: {desc}

Rules:
- Pandas imported as 'pd', dataframe is 'df', all strings lowercase
- Store answer in 'result'. No comments, no imports.
- Mix of simple retrieval, aggregation, and complex multi-step analysis
- Questions should be diverse and cover different columns

Return ONLY a JSON array: [{{"question": "...", "query": "result = ..."}}]"""

        print(f"  {role}...", end=" ", flush=True)
        start = time.time()

        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

        try:
            pairs = json.loads(raw)
        except json.JSONDecodeError:
            print(f"PARSE ERROR")
            pairs = []

        for p in pairs:
            p["role"] = role
        all_pairs.extend(pairs)

        elapsed = time.time() - start
        print(f"{len(pairs)} pairs ({elapsed:.1f}s)")

    # Save
    os.makedirs("results", exist_ok=True)
    with open(TRAINING_DATA_PATH, "w") as f:
        json.dump({"schema": SCHEMA, "pairs": all_pairs}, f, indent=2)

    print(f"\n  Total: {len(all_pairs)} pairs saved to {TRAINING_DATA_PATH}")
    return all_pairs


# ============================================================================
# Step 2: Build retrieval index
# ============================================================================

class QueryRetriever:
    """TF-IDF based retrieval over generated training pairs."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.questions = [p["question"] for p in pairs]
        self.queries = [p["query"] for p in pairs]

        # Build TF-IDF index
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams
            max_features=5000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
        print(f"  Built retrieval index: {len(self.questions)} questions, {self.tfidf_matrix.shape[1]} features")

    def retrieve(self, question, top_k=3):
        """Find the most similar training questions."""
        query_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "question": self.questions[idx],
                "query": self.queries[idx],
                "similarity": round(float(similarities[idx]), 3),
                "role": self.pairs[idx].get("role", "unknown"),
            })
        return results


# ============================================================================
# Step 3: Adapt retrieved query for new question
# ============================================================================

def adapt_query(new_question, retrieved_examples):
    """
    Use a lightweight Claude call to adapt a retrieved query template.
    
    Key: we send the TEMPLATE query + schema metadata + new question.
    We still never send any actual data.
    """
    examples_text = ""
    for i, ex in enumerate(retrieved_examples[:2]):  # top 2 examples
        examples_text += f"\nExample {i+1}:\n  Question: {ex['question']}\n  Query: {ex['query']}\n"

    col_info = ", ".join(f"{c} ({t})" for c, t in zip(SCHEMA["columns"], SCHEMA["dtypes"]))

    prompt = f"""Given these similar question/query examples from a heart disease dataframe:
{examples_text}
Dataframe columns: {col_info}

Write a Pandas query for this NEW question: {new_question}

Rules: Store answer in 'result'. No comments, no imports. Pandas is 'pd', dataframe is 'df'. All strings are lowercase.
Return ONLY the code."""

    message = client.messages.create(
        model=MODEL,
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(?:python)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)

    return raw.strip(), message.usage


# ============================================================================
# Step 4: Execute in sandbox
# ============================================================================

def execute_query(code, df):
    """Run generated code in restricted environment."""
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


# ============================================================================
# Step 5: Full pipeline
# ============================================================================

def run_pipeline(question, retriever, df):
    """
    Complete retrieval-augmented DataFrame QA pipeline.
    
    Returns dict with all intermediate steps for transparency.
    """
    pipeline_start = time.time()

    # Retrieve similar examples
    retrieved = retriever.retrieve(question, top_k=3)

    # Adapt query using templates
    adapted_code, usage = adapt_query(question, retrieved)

    # Execute
    output = execute_query(adapted_code, df)

    total_time = time.time() - pipeline_start

    return {
        "question": question,
        "retrieved_examples": retrieved,
        "generated_code": adapted_code,
        "execution": output,
        "tokens_used": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": usage.input_tokens + usage.output_tokens,
        },
        "time_seconds": round(total_time, 2),
    }


# ============================================================================
# Demo questions
# ============================================================================

DEMO_QUESTIONS = [
    "What is the average age of patients diagnosed with heart disease?",
    "How many female patients have cholesterol above 250?",
    "What percentage of patients with asymptomatic chest pain have heart disease?",
    "Which chest pain type has the highest average max heart rate?",
    "What is the correlation between age and cholesterol?",
    "How does resting blood pressure compare between patients with and without exercise angina?",
    "What are the top 3 most common thalassemia types?",
    "For patients over 60, what is the average number of major vessels?",
]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Retrieval-based DataFrame QA Demo")
    parser.add_argument("--skip-generation", action="store_true", help="Use cached training data")
    parser.add_argument("--interactive", action="store_true", help="Interactive question mode")
    parser.add_argument("--pairs-per-role", type=int, default=10, help="Training pairs per role")
    args = parser.parse_args()

    # Load sample data (stays local)
    df = pd.DataFrame(SAMPLE_DATA)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.lower()

    print(f"\n{'='*64}")
    print(f"  DataFrame QA — Retrieval-Based Demo")
    print(f"  Domain: Heart Disease ({len(df)} patients, {len(df.columns)} features)")
    print(f"  Data: NEVER sent to API (stays local)")
    print(f"{'='*64}")

    # Step 1: Generate or load training data
    if args.skip_generation and os.path.exists(TRAINING_DATA_PATH):
        print(f"\n  Loading cached training data from {TRAINING_DATA_PATH}")
        with open(TRAINING_DATA_PATH) as f:
            data = json.load(f)
        pairs = data["pairs"]
        print(f"  Loaded {len(pairs)} pairs")
    else:
        pairs = generate_training_pairs(num_per_role=args.pairs_per_role)

    # Step 2: Build retrieval index
    print(f"\n  Building retrieval index...")
    retriever = QueryRetriever(pairs)

    # Step 3: Run demo questions
    if args.interactive:
        print(f"\n{'='*64}")
        print(f"  Interactive mode — type a question or 'quit' to exit")
        print(f"{'='*64}\n")

        while True:
            question = input("  Your question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            result = run_pipeline(question, retriever, df)
            print_result(result)
    else:
        print(f"\n{'='*64}")
        print(f"  Running {len(DEMO_QUESTIONS)} demo questions")
        print(f"{'='*64}\n")

        total_tokens = 0
        successes = 0

        for i, question in enumerate(DEMO_QUESTIONS):
            print(f"  [{i+1}/{len(DEMO_QUESTIONS)}] {question}")

            result = run_pipeline(question, retriever, df)
            total_tokens += result["tokens_used"]["total"]

            if result["execution"]["success"]:
                successes += 1

            print_result(result)

        # Summary
        print(f"\n{'='*64}")
        print(f"  DEMO SUMMARY")
        print(f"{'='*64}")
        print(f"  Questions:       {len(DEMO_QUESTIONS)}")
        print(f"  Successful:      {successes}/{len(DEMO_QUESTIONS)}")
        print(f"  Total tokens:    {total_tokens:,}")
        print(f"  Avg tokens/q:    {total_tokens // len(DEMO_QUESTIONS)}")
        print(f"  Training pairs:  {len(pairs)} (generated from metadata)")
        print(f"  Data exposed:    0 rows (all execution local)")
        print(f"{'='*64}\n")

        # Save results
        os.makedirs("results", exist_ok=True)
        with open("results/retrieval_demo_results.json", "w") as f:
            json.dump({
                "questions": len(DEMO_QUESTIONS),
                "successes": successes,
                "total_tokens": total_tokens,
                "training_pairs": len(pairs),
            }, f, indent=2)
        print(f"  Results saved to results/retrieval_demo_results.json")


def print_result(result):
    """Pretty-print a pipeline result."""
    # Show top retrieved example
    top = result["retrieved_examples"][0]
    print(f"  ┌─ Retrieved: \"{top['question'][:60]}...\"")
    print(f"  │  Similarity: {top['similarity']:.1%}  |  Role: {top['role']}")
    print(f"  ├─ Generated: {result['generated_code']}")

    if result["execution"]["success"]:
        res = result["execution"]["result"]
        # Format result nicely
        if isinstance(res, (pd.Series, pd.DataFrame)):
            res_str = str(res)
            if len(res_str) > 200:
                res_str = res_str[:200] + "..."
        else:
            res_str = str(res)
        print(f"  ├─ Result: {res_str}")
    else:
        print(f"  ├─ ERROR: {result['execution']['error']}")

    print(f"  └─ Tokens: {result['tokens_used']['total']} | Time: {result['time_seconds']}s\n")


if __name__ == "__main__":
    main()
