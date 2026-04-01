"""
Evaluation pipeline for DataFrame QA.

Runs a set of question/answer pairs through the framework and computes
pass@1 accuracy, matching the paper's evaluation methodology.

Usage:
    python src/evaluate.py --dataset data/sample_qa.json --data data/sample.csv
    python src/evaluate.py --dataset data/sample_qa.json --data data/sample.csv --model claude-opus-4-6
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataframe_qa import DataFrameQA


# ---------------------------------------------------------------------------
# Result comparison (relaxed matching from the paper)
# ---------------------------------------------------------------------------

def normalize_value(val):
    """Normalize a value for comparison."""
    if val is None:
        return None
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return val.values.flatten().tolist()
    if isinstance(val, np.ndarray):
        return val.flatten().tolist()
    return val


def compare_results(predicted, expected) -> bool:
    """
    Relaxed comparison matching the paper's evaluation criteria.

    Handles:
      - Numeric comparison with tolerance
      - Case-insensitive string matching
      - Series/DataFrame containment checks
      - List/array matching
    """
    pred = normalize_value(predicted)
    exp = normalize_value(expected)

    # Both None
    if pred is None and exp is None:
        return True
    if pred is None or exp is None:
        return False

    # If predicted is a list (from Series/DataFrame), check containment
    if isinstance(pred, list):
        str_pred = [str(v).lower().strip() for v in pred]
        str_exp = str(exp).lower().strip()
        # Direct containment
        if str_exp in str_pred:
            return True
        # Try numeric containment
        try:
            float_exp = float(exp)
            for v in pred:
                try:
                    if abs(float(v) - float_exp) < 1e-6:
                        return True
                except (ValueError, TypeError):
                    continue
        except (ValueError, TypeError):
            pass
        return False

    # String comparison
    if isinstance(pred, str) or isinstance(exp, str):
        return str(pred).lower().strip() == str(exp).lower().strip()

    # Numeric comparison
    try:
        return abs(float(pred) - float(exp)) < 1e-6
    except (ValueError, TypeError):
        return str(pred) == str(exp)


# ---------------------------------------------------------------------------
# Error classification (8 types from the paper)
# ---------------------------------------------------------------------------

ERROR_TYPES = {
    "string_error": "String Matching and Comparison Errors",
    "access_error": "Data Access and Bounds Errors",
    "condition_error": "Query Condition and Value Errors",
    "type_error": "Data Type and Operation Errors",
    "expectation_error": "Expectation and Interpretation Errors",
    "structure_error": "Data Structure Reference Errors",
    "function_error": "Function and Method Usage Errors",
    "other": "Other Errors",
}


def classify_error(error_msg: str, code: str) -> list[str]:
    """Basic heuristic error classification based on the paper's taxonomy."""
    classes = []
    error_lower = (error_msg or "").lower()
    code_lower = (code or "").lower()

    if "keyerror" in error_lower or "not in index" in error_lower:
        classes.append("structure_error")
    if "indexerror" in error_lower or "out of bounds" in error_lower:
        classes.append("access_error")
    if "typeerror" in error_lower or "cannot convert" in error_lower:
        classes.append("type_error")
    if "attributeerror" in error_lower:
        classes.append("function_error")
    if "syntaxerror" in error_lower:
        classes.append("other")
    if ".upper()" in code_lower or ".title()" in code_lower:
        classes.append("string_error")
    if "import " in code_lower:
        classes.append("other")  # instruction misalignment

    return classes if classes else ["condition_error"]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    df: pd.DataFrame,
    qa_pairs: list[dict],
    model: str = "claude-sonnet-4-20250514",
    output_dir: str = "results",
) -> dict:
    """
    Run the full evaluation pipeline.

    Args:
        df: The dataframe to query
        qa_pairs: List of dicts with 'question' and 'expected_answer' keys
                  (optionally 'ground_truth_query' and 'column_descriptions')
        model: Claude model ID to use
        output_dir: Directory for saving results

    Returns:
        dict with 'pass_at_1', 'total', 'correct', 'errors'
    """
    qa = DataFrameQA(model=model, temperature=0)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    correct = 0
    results = []

    print(f"\nRunning DataFrame QA evaluation")
    print(f"  Model:     {model}")
    print(f"  Questions: {len(qa_pairs)}")
    print(f"  Columns:   {df.columns.tolist()}\n")

    for i, pair in enumerate(tqdm(qa_pairs, desc="Evaluating")):
        question = pair["question"]
        expected = pair["expected_answer"]
        col_descs = pair.get("column_descriptions")

        # Generate and execute
        start = time.time()
        output = qa.ask(df, question, column_descriptions=col_descs)
        elapsed = time.time() - start

        # Compare
        is_correct = False
        if output["success"]:
            is_correct = compare_results(output["result"], expected)

        if is_correct:
            correct += 1

        # Classify errors
        error_classes = []
        if not is_correct and output["error"]:
            error_classes = classify_error(output["error"], output["code"])

        results.append(
            {
                "index": i,
                "question": question,
                "generated_code": output["code"],
                "success": output["success"],
                "result": str(output["result"]),
                "expected": str(expected),
                "correct": is_correct,
                "error": output["error"],
                "error_classes": error_classes,
                "time_seconds": round(elapsed, 2),
            }
        )

    # Compute metrics
    total = len(qa_pairs)
    pass_at_1 = correct / total if total > 0 else 0

    # Summary
    print(f"\n{'='*50}")
    print(f"  pass@1: {pass_at_1:.1%} ({correct}/{total})")
    print(f"{'='*50}\n")

    # Save detailed results
    results_path = Path(output_dir) / f"eval_{model.replace('/', '_')}_{int(time.time())}.json"
    summary = {
        "model": model,
        "pass_at_1": pass_at_1,
        "total": total,
        "correct": correct,
        "results": results,
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved to {results_path}")

    # Print error breakdown
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\nError breakdown ({len(errors)} failures):")
        error_counts = {}
        for r in errors:
            for ec in r.get("error_classes", ["other"]):
                error_counts[ec] = error_counts.get(ec, 0) + 1
        for ec, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            label = ERROR_TYPES.get(ec, ec)
            print(f"  {label}: {count}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DataFrame QA")
    parser.add_argument("--dataset", required=True, help="Path to QA pairs JSON file")
    parser.add_argument("--data", required=True, help="Path to CSV dataframe")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model ID")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    with open(args.dataset) as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns, {len(qa_pairs)} questions")

    # Run
    run_evaluation(df, qa_pairs, model=args.model, output_dir=args.output)


if __name__ == "__main__":
    main()
