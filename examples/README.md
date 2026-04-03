# Examples

Runnable scripts demonstrating the DataFrame QA framework. All scripts should be run from the **project root** directory.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Scripts

### `api_walkthrough.py`
Full end-to-end walkthrough of the DataFrame QA pipeline. Generates Pandas queries from metadata, executes them in a sandbox, and evaluates pass@1 accuracy.

```bash
python examples/api_walkthrough.py
```

### `token_comparison.py`
Runs each question two ways — metadata-only vs full-table — and compares token usage, cost, and accuracy side by side. Demonstrates the ~90% token reduction the paper claims.

```bash
python examples/token_comparison.py
python examples/token_comparison.py --data data/sample.csv --questions data/sample_qa.json
```

### `generate_training_data.py`
Generates synthetic question/Pandas-query training pairs from schema metadata alone. Supports three user roles from the paper (Data Scientist, General User, Data Owner) and can validate generated pairs against real data.

```bash
# From a CSV file
python examples/generate_training_data.py --validate

# Schema-only mode (no data file needed)
python examples/generate_training_data.py --schema-only \
  --columns "age,sex,bp,cholesterol,target" \
  --dtypes "int,str,float,float,int" \
  --dataset-name "Heart Disease" \
  --pairs-per-role 10
```

### `cross_domain_gen.py`
Generates training data across three domains (medical, automotive, sports) to demonstrate cross-domain scalability. Uses only column metadata — no real data.

```bash
python examples/cross_domain_gen.py
```

## Output

All results are saved to the `results/` directory as JSON files.
