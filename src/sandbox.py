"""
Restricted execution environment for generated Pandas queries.

Only allows pandas, numpy, and math — matching the paper's sandbox constraints.
"""

import pandas as pd
import numpy as np
import math


class SafeExecutor:
    """Execute LLM-generated Pandas code in a restricted namespace."""

    ALLOWED_MODULES = {"pd": pd, "np": np, "math": math}

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str, df: pd.DataFrame) -> dict:
        """
        Execute code with only the dataframe and safe libraries available.

        Returns:
            dict with keys:
                - 'success': bool
                - 'result': the value of the 'result' variable, or None
                - 'error': error message if failed, or None
        """
        safe_globals = {
            **self.ALLOWED_MODULES,
            "df": df.copy(),  # work on a copy to prevent mutation
            "__builtins__": {},
        }
        local_vars = {}

        try:
            exec(code, safe_globals, local_vars)
            result = local_vars.get("result")
            if result is None and "result" not in local_vars:
                return {
                    "success": False,
                    "result": None,
                    "error": "No 'result' variable found in generated code",
                }
            return {"success": True, "result": result, "error": None}

        except SyntaxError as e:
            return {
                "success": False,
                "result": None,
                "error": f"SyntaxError: {e}",
            }
        except NameError as e:
            return {
                "success": False,
                "result": None,
                "error": f"NameError (possibly blocked import): {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"{type(e).__name__}: {e}",
            }
