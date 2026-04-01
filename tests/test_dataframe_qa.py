"""Unit tests for DataFrame QA components."""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sandbox import SafeExecutor


# ---------------------------------------------------------------------------
# Sandbox tests
# ---------------------------------------------------------------------------

@pytest.fixture
def executor():
    return SafeExecutor()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "player": ["jalen rose", "terrence ross", "shawn respert"],
            "nationality": ["united states", "united states", "united states"],
            "position": ["guard-forward", "guard-forward", "guard"],
        }
    )


def test_sandbox_basic_query(executor, sample_df):
    code = "result = df[df['player'] == 'jalen rose']['position'].values[0]"
    output = executor.execute(code, sample_df)
    assert output["success"] is True
    assert output["result"] == "guard-forward"


def test_sandbox_aggregation(executor, sample_df):
    code = "result = df.shape[0]"
    output = executor.execute(code, sample_df)
    assert output["success"] is True
    assert output["result"] == 3


def test_sandbox_blocks_imports(executor, sample_df):
    code = "import os; result = os.getcwd()"
    output = executor.execute(code, sample_df)
    assert output["success"] is False
    assert "NameError" in output["error"]


def test_sandbox_no_result_var(executor, sample_df):
    code = "x = df.shape[0]"
    output = executor.execute(code, sample_df)
    assert output["success"] is False
    assert "No 'result' variable" in output["error"]


def test_sandbox_does_not_mutate_original(executor, sample_df):
    original_len = len(sample_df)
    code = "df.drop(df.index, inplace=True); result = df.shape[0]"
    executor.execute(code, sample_df)
    assert len(sample_df) == original_len  # original unchanged


# ---------------------------------------------------------------------------
# Comparison tests (import from evaluate)
# ---------------------------------------------------------------------------

from evaluate import compare_results


def test_compare_exact_string():
    assert compare_results("united states", "united states") is True


def test_compare_case_insensitive():
    assert compare_results("United States", "united states") is True


def test_compare_numeric():
    assert compare_results(6, 6) is True
    assert compare_results(6.0, 6) is True


def test_compare_numeric_tolerance():
    assert compare_results(3.1415926, 3.1415926) is True


def test_compare_series_containment():
    series = pd.Series(["jalen rose", "terrence ross"])
    assert compare_results(series, "jalen rose") is True


def test_compare_mismatch():
    assert compare_results("canada", "united states") is False
    assert compare_results(5, 6) is False
