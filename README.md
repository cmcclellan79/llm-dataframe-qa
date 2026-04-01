# LLM DataFrame QA

A Python implementation of the [DataFrame QA framework](https://arxiv.org/abs/2401.15463) (Ye et al., 2024), which uses large language models to generate Pandas queries for dataframe question answering **without exposing the underlying data**.

The LLM receives only column names and data types — never the actual values — ensuring data privacy while keeping prompts under ~250 tokens.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Column Names   │     │  Claude API  │     │   Sandbox    │     │   Compare    │
│  + Data Types   │ ──▶ │  Generates   │ ──▶ │   Executes   │ ──▶ │   Against    │
│  + Question     │     │  Pandas Code │     │   Query      │     │  Ground Truth│
└─────────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
      Prompt              LLM Response          Execution            Evaluation
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

### 3. Run a single query

```python
from src.dataframe_qa import DataFrameQA
import pandas as pd

df = pd.read_csv("your_data.csv")
qa = DataFrameQA(model="claude-sonnet-4-20250514")

result = qa.ask(df, "What is the average price by category?")
print(result)
```

### 4. Run the full evaluation

```bash
python src/evaluate.py --dataset data/sample_qa.json --data data/sample.csv
```

## Project Structure

```
dataframe-qa/
├── src/
│   ├── __init__.py
│   ├── dataframe_qa.py      # Core framework: prompt building, API calls, execution
│   ├── evaluate.py           # Evaluation pipeline with pass@1 scoring
│   └── sandbox.py            # Restricted code execution environment
├── data/
│   ├── sample.csv            # Example dataset for testing
│   └── sample_qa.json        # Example question/answer pairs
├── tests/
│   └── test_dataframe_qa.py  # Unit tests
├── results/                  # Evaluation output directory
├── requirements.txt
├── .env.example
├── .gitignore
└── LICENSE
```

## Features

- **Privacy-preserving**: Only schema metadata is sent to the LLM — data never leaves your machine
- **Zero-shot**: No fine-tuning or training required
- **Model-agnostic**: Swap between Claude Opus 4.6, Sonnet 4.6, or Haiku 4.5
- **Sandboxed execution**: Generated code runs in a restricted environment (Pandas, NumPy, Math only)
- **Evaluation toolkit**: Built-in pass@1 scoring with type-aware result comparison
- **Error classification**: Automatic categorization of failures (8 error types from the paper)

## Supported Models

| Model | ID | Best For |
|-------|-----|----------|
| Claude Opus 4.6 | `claude-opus-4-6` | Highest accuracy |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | Best cost/accuracy balance |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Budget / high-volume runs |

## How It Works

The framework follows the paper's three-stage pipeline:

1. **Pandas Query Generation** — The LLM receives a system prompt with constraints, the dataframe's column names and data types, and the user's natural language question. It returns executable Pandas code.

2. **Code Execution** — The generated code runs in a sandboxed environment restricted to `pandas`, `numpy`, and `math`. Results are captured as Python objects.

3. **Result Evaluation** — Executed results are compared against ground truth using relaxed matching that handles type differences (numeric tolerance, case-insensitive strings, Series/DataFrame containment checks).

## Configuration

You can customize the system prompt, execution constraints, and evaluation behavior:

```python
qa = DataFrameQA(
    model="claude-opus-4-6",     # Which Claude model to use
    temperature=0,                # Greedy decoding (deterministic)
    max_tokens=512,               # Max response length
    lowercase_strings=True,       # Standardize strings to lowercase
)
```

## References

- **Paper**: Ye, J., Du, M., & Wang, G. (2024). *DataFrame QA: A Universal LLM Framework on DataFrame Question Answering Without Data Exposure*. [arXiv:2401.15463](https://arxiv.org/abs/2401.15463)
- **WikiSQL**: [github.com/salesforce/WikiSQL](https://github.com/salesforce/WikiSQL)
- **Claude API Docs**: [platform.claude.com/docs](https://platform.claude.com/docs)

## License

MIT
