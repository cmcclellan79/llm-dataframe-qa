"""
DataFrame QA: Generate and execute Pandas queries using the Claude API.

Replicates the framework from:
  Ye, Du & Wang (2024) — "DataFrame QA: A Universal LLM Framework
  on DataFrame Question Answering Without Data Exposure"
  https://arxiv.org/abs/2401.15463
"""

import os
import re
from typing import Optional

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from .sandbox import SafeExecutor

load_dotenv()


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


class DataFrameQA:
    """
    Privacy-preserving DataFrame question answering via LLM-generated Pandas queries.

    Only column names and data types are sent to the API — the actual data
    never leaves your machine.

    Usage:
        qa = DataFrameQA(model="claude-sonnet-4-20250514")
        result = qa.ask(df, "What is the average price by category?")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0,
        max_tokens: int = 512,
        lowercase_strings: bool = True,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lowercase_strings = lowercase_strings
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.executor = SafeExecutor()

    def _extract_metadata(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Extract column names and data types from a dataframe."""
        columns = df.columns.tolist()
        dtypes = [str(dt) for dt in df.dtypes.tolist()]
        return columns, dtypes

    def _build_user_prompt(
        self,
        columns: list[str],
        dtypes: list[str],
        question: str,
        column_descriptions: Optional[dict[str, str]] = None,
    ) -> str:
        """Build the user-facing prompt with metadata and question."""
        prompt = f"""You are given a Pandas dataframe named 'df':
- Columns: {columns}
- Data Types: {dtypes}"""

        if column_descriptions:
            for col, desc in column_descriptions.items():
                prompt += f"\n- Column '{col}': {desc}"

        prompt += f"\n- User's Question: {question}"
        return prompt

    def _clean_code(self, raw: str) -> str:
        """Strip markdown fences and leading/trailing whitespace."""
        code = raw.strip()
        # Remove ```python ... ``` wrappers
        if code.startswith("```"):
            code = re.sub(r"^```(?:python)?\s*\n?", "", code)
            code = re.sub(r"\n?```\s*$", "", code)
        # Remove [PYTHON] ... [/PYTHON] wrappers (paper format)
        code = re.sub(r"^\[PYTHON\]\s*\n?", "", code)
        code = re.sub(r"\n?\[/PYTHON\]\s*$", "", code)
        return code.strip()

    def generate_query(
        self,
        df: pd.DataFrame,
        question: str,
        column_descriptions: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Generate a Pandas query for the given question using only schema metadata.

        Args:
            df: The dataframe (only metadata is sent to the API)
            question: Natural language question
            column_descriptions: Optional dict mapping column names to descriptions

        Returns:
            Generated Python code as a string
        """
        columns, dtypes = self._extract_metadata(df)

        user_prompt = self._build_user_prompt(
            columns, dtypes, question, column_descriptions
        )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_code = message.content[0].text
        return self._clean_code(raw_code)

    def ask(
        self,
        df: pd.DataFrame,
        question: str,
        column_descriptions: Optional[dict[str, str]] = None,
    ) -> dict:
        """
        End-to-end: generate a query, execute it, and return the result.

        Args:
            df: The dataframe to query
            question: Natural language question
            column_descriptions: Optional column descriptions for disambiguation

        Returns:
            dict with keys: 'question', 'code', 'result', 'success', 'error'
        """
        working_df = df.copy()
        if self.lowercase_strings:
            for col in working_df.select_dtypes(include=["object"]).columns:
                working_df[col] = working_df[col].str.lower()

        code = self.generate_query(working_df, question, column_descriptions)
        execution = self.executor.execute(code, working_df)

        return {
            "question": question,
            "code": code,
            "result": execution["result"],
            "success": execution["success"],
            "error": execution["error"],
        }
