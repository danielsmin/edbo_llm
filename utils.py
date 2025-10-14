"""Chatbot tools registry (template)

This module provides a small, dependency-light registry for functions the
chatbot can call as "tools". You can register functions with a decorator,
inspect available tools, and invoke a tool by name with validated kwargs.

How to add a new tool:

1) Define a normal Python function with keyword arguments.
2) Decorate it with @register_tool, providing a short description and optional
   input/output schemas (free-form JSON to help UIs or the LLM).
3) The function should return JSON-serializable data (dict/list/str/number).
   If it returns other objects (e.g., pandas/numpy), we'll try to coerce them
   via ensure_json_safe; fallback is str(...).

Example (uncomment to use):

    # @register_tool(
    #     description="Echo back the provided text",
    #     input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    #     output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    # )
    # def echo(text: str) -> dict:
    #     return {"text": text}

Runtime API:
- list_tools() -> List[dict]
- get_tool(name) -> tuple(callable, meta) | (None, None)
- call_tool(name, **kwargs) -> ToolResult

This file is a template—feel free to add your domain-specific tools here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import inspect


# ---------------- Result wrapper ---------------- #


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "data": ensure_json_safe(self.data),
            "error": self.error,
            "meta": self.meta or {},
        }


# ---------------- Registry ---------------- #


_TOOLS: Dict[str, Callable[..., Any]] = {}
_META: Dict[str, Dict[str, Any]] = {}


def register_tool(
    name: Optional[str] = None,
    *,
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
):
    """Decorator to register a function as a chatbot-callable tool.

    The function should accept keyword arguments only (recommended), or be
    defined to accept **kwargs. Positional-only parameters are not supported.
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        if tool_name in _TOOLS:
            raise ValueError(f"Tool already registered: {tool_name}")
        _TOOLS[tool_name] = func
        _META[tool_name] = {
            "name": tool_name,
            "description": description.strip(),
            "input_schema": input_schema or {},
            "output_schema": output_schema or {},
        }
        return func

    return _decorator


def list_tools() -> List[Dict[str, Any]]:
    """Return metadata for all registered tools."""
    # Ensure stable order for display
    names = sorted(_TOOLS.keys())
    return [dict(_META[n]) for n in names]


def get_tool(name: str) -> Tuple[Optional[Callable[..., Any]], Optional[Dict[str, Any]]]:
    """Return (callable, meta) for a tool name, or (None, None) if missing."""
    return _TOOLS.get(name), _META.get(name)


def call_tool(name: str, /, **kwargs: Any) -> ToolResult:
    """Invoke a registered tool by name with keyword args.

    - Validates unknown and missing required parameters against the function's
      signature (for POSITIONAL_OR_KEYWORD and KEYWORD_ONLY params).
    - Returns ToolResult with JSON-safe data.
    """
    func = _TOOLS.get(name)
    if func is None:
        return ToolResult(ok=False, error=f"Unknown tool: {name}")

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        sig = None  # builtins or C extensions

    # Validate kwargs against signature when available
    if sig is not None:
        params = sig.parameters
        # unknown args
        unknown = [k for k in kwargs.keys() if k not in params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())]
        if unknown:
            return ToolResult(ok=False, error=f"Unknown argument(s) for {name}: {', '.join(unknown)}")
        # missing required
        required = [
            p.name
            for p in params.values()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and p.default is inspect._empty
            and p.kind != inspect.Parameter.VAR_POSITIONAL
        ]
        missing = [r for r in required if r not in kwargs]
        if missing:
            return ToolResult(ok=False, error=f"Missing required argument(s) for {name}: {', '.join(missing)}")

    try:
        result = func(**kwargs)
        return ToolResult(ok=True, data=ensure_json_safe(result))
    except Exception as e:
        return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")


# ---------------- JSON safety helpers ---------------- #


def ensure_json_safe(obj: Any) -> Any:
    """Best-effort conversion of common Python/scientific types to JSON-safe.

    - Path -> str
    - set/tuple -> list
    - dataclass -> dict
    - pandas DataFrame/Series -> to_dict
    - numpy scalars/arrays -> tolist/py scalar
    - other unknowns -> str(obj)
    """
    # None, bool, int, float, str, dict, list already JSON-friendly (assuming nested ok)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return [ensure_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [ensure_json_safe(x) for x in obj]
    if is_dataclass(obj):
        try:
            return ensure_json_safe(asdict(obj))
        except Exception:
            return str(obj)
    # pandas
    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
    except Exception:
        pass
    # numpy
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy scalar types
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass
    # dict/list: recurse
    if isinstance(obj, dict):
        return {str(k): ensure_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ensure_json_safe(x) for x in obj]
    # fallback
    return str(obj)


__all__ = [
    "ToolResult",
    "register_tool",
    "list_tools",
    "get_tool",
    "call_tool",
    "ensure_json_safe",
]


# ---------------- Example tool (one only) ---------------- #


@register_tool(
    description="Echo back the provided text",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    output_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
    },
)
def echo(text: str) -> dict:
    """Return the provided text in a JSON object.

    Useful as a template for adding new chatbot-callable tools.
    """
    return {"text": str(text)}


# ---------------- Domain tool: predictions summary ---------------- #


@register_tool(
    description=(
        "Given a predictions CSV, find the overall best entries for: "
        "(1) max *_predicted_mean, (2) max *_predicted_variance, and (3) max *_expected_improvement."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string", "description": "Path to the predictions CSV"}
        },
        "required": ["csv_path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "best_predicted_mean": {
                "type": "object",
                "properties": {"column": {"type": "string"}, "value": {}, "row_index": {"type": "integer"}},
            },
            "best_predicted_variance": {
                "type": "object",
                "properties": {"column": {"type": "string"}, "value": {}, "row_index": {"type": "integer"}},
            },
            "best_expected_improvement": {
                "type": "object",
                "properties": {"column": {"type": "string"}, "value": {}, "row_index": {"type": "integer"}},
            },
        },
    },
)
def best_predictions(csv_path: str) -> dict:
    """Return best predicted mean (max), variance (max), and expected improvement (max) from a CSV.

    This scans columns whose names end with these suffixes:
    - _predicted_mean
    - _predicted_variance
    - _expected_improvement

    It aggregates the global best row across all columns matching each suffix and
    returns the column name, the value at that row, and the row index.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pandas is required for best_predictions: {e}")

    df = pd.read_csv(csv_path)

    def _best_for_suffix(suffix: str, prefer: str) -> dict:
        # prefer: 'max' or 'min'
        cols = [c for c in df.columns if isinstance(c, str) and c.endswith(suffix)]
        best = {"column": None, "value": None, "row_index": None}
        if not cols:
            return best
        best_val = None
        best_idx = None
        best_col = None
        for c in cols:
            series = df[c]
            # drop NaNs
            valid = series.dropna()
            if valid.empty:
                continue
            if prefer == "max":
                idx = valid.idxmax()
                val = valid.loc[idx]
            else:
                idx = valid.idxmin()
                val = valid.loc[idx]
            if best_val is None:
                best_val, best_idx, best_col = val, idx, c
            else:
                if prefer == "max" and val > best_val:
                    best_val, best_idx, best_col = val, idx, c
                if prefer == "min" and val < best_val:
                    best_val, best_idx, best_col = val, idx, c
        if best_col is not None:
            best = {"column": best_col, "value": ensure_json_safe(best_val), "row_index": int(best_idx)}
        return best

    res = {
        "best_predicted_mean": _best_for_suffix("_predicted_mean", prefer="max"),
        "best_predicted_variance": _best_for_suffix("_predicted_variance", prefer="max"),
        "best_expected_improvement": _best_for_suffix("_expected_improvement", prefer="max"),
    }
    return res

