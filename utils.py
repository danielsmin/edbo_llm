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
import json


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
    # REPL helpers
    "CommandContext",
    "handle_repl_command",
    "_pretty_print_best_predictions",
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


# ---------------- Data helpers for additional tools ---------------- #


def _load_df(csv_path: str):
    import pandas as pd  # type: ignore
    df = pd.read_csv(csv_path)
    return df


def _ensure_artifacts_dir() -> Path:
    out_dir = Path.cwd() / ".edbo_llm_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_fig(plt, name: str) -> str:
    out_dir = _ensure_artifacts_dir()
    path = out_dir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


# ---------------- Additional analysis/visualization tools ---------------- #


@register_tool(
    description="List unique values and counts for a column.",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "column": {"type": "string"},
            "top_n": {"type": "integer"},
        },
        "required": ["csv_path", "column"],
    },
)
def feature_values(csv_path: str, column: str, top_n: int = 50) -> dict:
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")
    counts = df[column].value_counts(dropna=False).head(int(top_n))
    return {
        "column": column,
        "distinct": int(df[column].nunique(dropna=False)),
        "top": [{"value": (None if pd.isna(k) else ensure_json_safe(k)), "count": int(v)} for k, v in counts.items()],
    }


@register_tool(
    description="Summary stats for a numeric label column (min/mean/median/max/std/quantiles).",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["csv_path", "label"],
    },
)
def label_stats(csv_path: str, label: str) -> dict:
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if label not in df.columns:
        raise ValueError(f"Label not found: {label}")
    s = pd.to_numeric(df[label], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Label has no numeric values: {label}")
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    return {
        "label": label,
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "median": float(s.median()),
        "max": float(s.max()),
        "quantiles": {str(k): float(v) for k, v in q.items()},
    }


@register_tool(
    description="Histogram of a numeric label column; returns image path to PNG.",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "label": {"type": "string"},
            "bins": {"type": "integer"},
        },
        "required": ["csv_path", "label"],
    },
)
def plot_label_hist(csv_path: str, label: str, bins: int = 30) -> dict:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib is required: {e}")
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if label not in df.columns:
        raise ValueError(f"Label not found: {label}")
    s = pd.to_numeric(df[label], errors="coerce").dropna()
    fig = plt.figure(figsize=(6, 4))
    plt.hist(s, bins=int(bins), color="#4C78A8", edgecolor="white")
    plt.title(f"Histogram of {label}")
    plt.xlabel(label)
    plt.ylabel("count")
    path = _save_fig(plt, f"hist_{label}")
    return {"image_path": path, "label": label}


@register_tool(
    description="Plot feature vs label: scatter for numeric feature, boxplot for categorical. Returns image path.",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "feature": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["csv_path", "feature", "label"],
    },
)
def plot_feature_vs_label(csv_path: str, feature: str, label: str) -> dict:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib is required: {e}")
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if feature not in df.columns or label not in df.columns:
        raise ValueError("feature or label not found in CSV")
    f = df[feature]
    y = pd.to_numeric(df[label], errors="coerce")
    fig = plt.figure(figsize=(6, 4))
    if pd.api.types.is_numeric_dtype(f):
        plt.scatter(f, y, s=18, alpha=0.75, color="#4C78A8")
        plt.xlabel(feature)
        plt.ylabel(label)
        plt.title(f"{feature} vs {label}")
    else:
        # Boxplot per category (limit to first 30 categories)
        cats = f.astype(str).fillna("NA")
        order = cats.value_counts().index.tolist()[:30]
        data = [y[cats == c].dropna().values for c in order]
        plt.boxplot(data, labels=order, showmeans=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(label)
        plt.title(f"{feature} vs {label} (boxplot)")
    path = _save_fig(plt, f"feature_vs_label_{feature}_{label}")
    return {"image_path": path, "feature": feature, "label": label}


@register_tool(
    description="Correlation heatmap for numeric columns; returns image path and top correlated pairs.",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "top_k_pairs": {"type": "integer"},
        },
        "required": ["csv_path"],
    },
)
def correlation_heatmap(csv_path: str, columns: list | None = None, top_k_pairs: int = 10) -> dict:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib is required: {e}")
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if columns:
        use = [c for c in columns if c in df.columns]
    else:
        use = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not use:
        raise ValueError("No numeric columns to correlate.")
    corr = df[use].corr(numeric_only=True)
    fig = plt.figure(figsize=(max(6, 0.4*len(use)), max(5, 0.4*len(use))))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(use)), use, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(use)), use, fontsize=8)
    plt.title("Correlation heatmap")
    path = _save_fig(plt, "correlation_heatmap")
    # top pairs by absolute correlation (excluding diagonal)
    pairs = []
    for i, a in enumerate(use):
        for j, b in enumerate(use):
            if j <= i:
                continue
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            pairs.append({"a": a, "b": b, "corr": float(val), "abs_corr": float(abs(val))})
    pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
    return {"image_path": path, "top_pairs": pairs[: int(top_k_pairs)]}


@register_tool(
    description="Pareto front for two objectives (maximize both); returns image path and indices.",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "x_col": {"type": "string", "description": "e.g., yield_predicted_mean"},
            "y_col": {"type": "string", "description": "e.g., ee_predicted_mean"},
        },
        "required": ["csv_path", "x_col", "y_col"],
    },
)
def pareto_front(csv_path: str, x_col: str, y_col: str) -> dict:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib is required: {e}")
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("x_col or y_col not found in CSV")
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = (~x.isna()) & (~y.isna())
    X = x[valid].values
    Y = y[valid].values
    idxs = x[valid].index.tolist()
    # Compute Pareto (maximize both)
    order = sorted(range(len(X)), key=lambda i: (-X[i], -Y[i]))
    pareto = []
    best_y = float("-inf")
    for i in order:
        if Y[i] >= best_y:
            pareto.append(i)
            best_y = Y[i]
    pareto_indices = [int(idxs[i]) for i in pareto]
    # Plot
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(X, Y, s=14, alpha=0.5, color="#9ecae1", label="all")
    plt.scatter([X[i] for i in pareto], [Y[i] for i in pareto], s=24, color="#08519c", label="Pareto")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Pareto front (maximize both)")
    plt.legend()
    path = _save_fig(plt, f"pareto_{x_col}_{y_col}")
    return {"image_path": path, "pareto_indices": pareto_indices}


@register_tool(
    description="Top-N rows by a metric column (max or min).",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "metric_column": {"type": "string"},
            "n": {"type": "integer"},
            "prefer": {"type": "string", "enum": ["max", "min"]},
        },
        "required": ["csv_path", "metric_column"],
    },
)
def top_n_by_metric(csv_path: str, metric_column: str, n: int = 10, prefer: str = "max") -> dict:
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if metric_column not in df.columns:
        raise ValueError(f"Metric column not found: {metric_column}")
    s = pd.to_numeric(df[metric_column], errors="coerce")
    valid = s.dropna()
    if valid.empty:
        raise ValueError("Metric column has no numeric values")
    if prefer == "min":
        order = valid.sort_values(ascending=True).head(int(n))
    else:
        order = valid.sort_values(ascending=False).head(int(n))
    rows = []
    for idx, val in order.items():
        rows.append({"row_index": int(idx), "value": float(val)})
    return {"metric": metric_column, "prefer": prefer, "rows": rows}


@register_tool(
    description="Group mean of a label by a categorical feature; returns table of category, count, mean.",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "feature": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["csv_path", "feature", "label"],
    },
)
def group_mean_label_by_feature(csv_path: str, feature: str, label: str) -> dict:
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if feature not in df.columns or label not in df.columns:
        raise ValueError("feature or label not found in CSV")
    y = pd.to_numeric(df[label], errors="coerce")
    cats = df[feature].astype(str).fillna("NA")
    g = (
        pd.DataFrame({"cat": cats, "y": y})
        .dropna(subset=["y"]) 
        .groupby("cat")
        .agg(count=("y", "count"), mean=("y", "mean"))
        .sort_values("mean", ascending=False)
    )
    out = [{"category": k, "count": int(v["count"]), "mean": float(v["mean"]) } for k, v in g.to_dict(orient="index").items()]
    return {"feature": feature, "label": label, "groups": out}


@register_tool(
    description="Describe a row's reaction conditions by index, excluding prediction columns by default.",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "row_index": {"type": "integer"},
            "exclude_suffixes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Column name suffixes to exclude",
            },
            "exclude_columns": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["csv_path", "row_index"],
    },
)
def describe_row(
    csv_path: str,
    row_index: int,
    exclude_suffixes: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> dict:
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if row_index not in df.index:
        raise ValueError(f"Row index not found: {row_index}")
    row = df.loc[row_index]
    if exclude_suffixes is None:
        exclude_suffixes = [
            "_predicted_mean",
            "_predicted_variance",
            "_expected_improvement",
        ]
    if exclude_columns is None:
        exclude_columns = []
    conditions = {}
    for c, v in row.items():
        if c in exclude_columns:
            continue
        if any(str(c).endswith(suf) for suf in exclude_suffixes):
            continue
        conditions[str(c)] = ensure_json_safe(v)
    return {"row_index": int(row_index), "conditions": conditions}


@register_tool(
    description="Compare two rows and report differing conditions; excludes prediction columns by default.",
    input_schema={
        "type": "object",
        "properties": {
            "csv_path": {"type": "string"},
            "row_a": {"type": "integer"},
            "row_b": {"type": "integer"},
        },
        "required": ["csv_path", "row_a", "row_b"],
    },
)
def compare_rows(csv_path: str, row_a: int, row_b: int) -> dict:
    import pandas as pd  # type: ignore
    df = _load_df(csv_path)
    if row_a not in df.index or row_b not in df.index:
        raise ValueError("row_a or row_b not in DataFrame index")
    ra = df.loc[row_a]
    rb = df.loc[row_b]
    exclude_suffixes = ["_predicted_mean", "_predicted_variance", "_expected_improvement"]
    diffs = []
    for c in df.columns:
        if any(str(c).endswith(suf) for suf in exclude_suffixes):
            continue
        va = ra[c]
        vb = rb[c]
        # Treat NaN equality
        try:
            na = pd.isna(va)
            nb = pd.isna(vb)
        except Exception:
            na = False
            nb = False
        equal = (na and nb) or (va == vb)
        if not equal:
            diffs.append({
                "column": str(c),
                "row_a": ensure_json_safe(va),
                "row_b": ensure_json_safe(vb),
            })
    return {"row_a": int(row_a), "row_b": int(row_b), "differences": diffs}


# ---------------- REPL command handlers (migrated from command_utils.py) ---------------- #


@dataclass
class CommandContext:
    space: Any  # SearchSpace
    cfg: Any    # ChatConfig
    descriptor_db: dict | None


def _csv_path_from_space(space) -> str:
    try:
        return str(Path(space.df.attrs.get('source', '') or '').resolve())
    except Exception:
        return ''


def _pretty_print_best_predictions(data: dict, df=None, label_cols=None) -> None:
    def _fmt_value(v):
        try:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return f"{v:.4f}"
            return str(v)
        except Exception:
            return str(v)

    def _condition_cols(_df):
        if _df is None:
            return []
        suffixes = ("_predicted_mean", "_predicted_variance", "_expected_improvement")
        cols = []
        for c in _df.columns:
            cs = str(c)
            if any(cs.endswith(s) for s in suffixes):
                continue
            if label_cols and c in label_cols:
                continue
            if cs.lower() == "index" or cs.startswith("Unnamed:"):
                continue
            cols.append(c)
        return cols

    def _fmt_row_conditions(_df, idx):
        try:
            if _df is None:
                return None
            row = _df.loc[idx] if idx in _df.index else _df.iloc[int(idx)]
            parts = []
            for c in _condition_cols(_df):
                try:
                    val = row[c]
                    sval = f"{val:.4f}" if isinstance(val, float) else str(val)
                except Exception:
                    sval = "?"
                parts.append(f"{c}={sval}")
            return ", ".join(parts) if parts else None
        except Exception:
            return None

    sections = [
        ("Best predicted mean", data.get("best_predicted_mean")),
        ("Best predicted variance (max)", data.get("best_predicted_variance")),
        ("Best expected improvement", data.get("best_expected_improvement")),
    ]
    for title, entry in sections:
        if not entry or not isinstance(entry, dict) or not entry.get("column"):
            print(f"- {title}: not found")
            continue
        val = _fmt_value(entry.get("value"))
        idx = entry.get("row_index")
        col = entry.get("column")
        print(f"- {title}: {val} (column: {col})")
        conds = _fmt_row_conditions(df, idx)
        if conds:
            print(f"  conditions: {conds}")


def cmd_help(ctx: CommandContext, user: str) -> None:
    print("Commands: :help, :features, :labels, :preview, :model, :describe <feature>, :tools, :call <tool> <json-kwargs>, :best-preds [csv_path], :plot_label_hist <label> [bins], :hist <label> [bins], :plot_feature_vs_label <feature> <label>, :corr_heatmap [col1,col2,...], :pareto <x_col> <y_col>")


def cmd_features(ctx: CommandContext, user: str) -> None:
    print("Features:", ", ".join(ctx.space.features))


def cmd_labels(ctx: CommandContext, user: str) -> None:
    print("Labels:", ", ".join(ctx.space.labels) if ctx.space.labels else "(none specified)")


def cmd_preview(ctx: CommandContext, user: str) -> None:
    from .search_space import preview_df
    print(preview_df(ctx.space.df, 5))


def cmd_model(ctx: CommandContext, user: str) -> None:
    print("Current model:", ctx.cfg.model_name)


def cmd_describe(ctx: CommandContext, user: str) -> None:
    parts = user.split(maxsplit=1)
    if len(parts) == 2:
        feat_name = parts[1].strip()
        if ctx.descriptor_db:
            d = ctx.descriptor_db.get(feat_name)
            print(d if d else f"No descriptor definition for: {feat_name}")
        else:
            print("Descriptor DB not loaded.")
    else:
        print("Usage: :describe <feature>")


def cmd_best_preds(ctx: CommandContext, user: str) -> None:
    parts = user.split(maxsplit=1)
    if len(parts) == 2:
        csv_path = parts[1].strip()
    else:
        csv_path = _csv_path_from_space(ctx.space)
    if not csv_path:
        print("No CSV path detected. Provide one: :best-preds /path/to/predictions.csv")
        return
    result = call_tool("best_predictions", csv_path=csv_path)
    if result.ok and isinstance(result.data, dict):
        _pretty_print_best_predictions(result.data, df=ctx.space.df, label_cols=ctx.space.labels)
    else:
        try:
            print(json.dumps(result.to_json(), indent=2))
        except Exception:
            print(result.to_json())


def cmd_tools(ctx: CommandContext, user: str) -> None:
    tools = list_tools()
    if not tools:
        print("No tools registered.")
    else:
        for t in tools:
            name = t.get("name", "?")
            desc = t.get("description", "")
            print(f"- {name}: {desc}")


def cmd_call(ctx: CommandContext, user: str) -> None:
    parts = user.split(maxsplit=2)
    if len(parts) < 2:
        print("Usage: :call <tool> <json-kwargs>")
        return
    tool_name = parts[1]
    kwargs = {}
    if len(parts) == 3:
        json_arg = parts[2]
        try:
            kwargs = json.loads(json_arg)
            if not isinstance(kwargs, dict):
                print("Error: JSON kwargs must be an object, e.g., {\"csv_path\": \"/path/file.csv\"}")
                return
        except Exception as e:
            print(f"JSON parse error: {e}")
            return
    result = call_tool(tool_name, **kwargs)
    if tool_name == "best_predictions" and result.ok and isinstance(result.data, dict):
        src = ctx.space.df.attrs.get('source') if getattr(ctx.space, 'df', None) is not None else None
        csv_kw = kwargs.get('csv_path') if isinstance(kwargs, dict) else None
        df_for_print = None
        try:
            if src and csv_kw and Path(src).resolve() == Path(csv_kw).resolve():
                df_for_print = ctx.space.df
        except Exception:
            df_for_print = None
        _pretty_print_best_predictions(result.data, df=df_for_print, label_cols=(ctx.space.labels if df_for_print is not None else None))
    else:
        try:
            print(json.dumps(result.to_json(), indent=2))
        except Exception:
            print(result.to_json())


def cmd_plot_label_hist(ctx: CommandContext, user: str) -> None:
    parts = user.split()
    if len(parts) < 2:
        print("Usage: :plot_label_hist <label> [bins]")
        return
    label = parts[1]
    bins = 30
    if len(parts) >= 3:
        try:
            bins = int(parts[2])
        except Exception:
            print("bins must be an integer; using default 30")
            bins = 30
    csv_path = _csv_path_from_space(ctx.space)
    if not csv_path:
        print("No CSV path detected. Ensure you launched with --scope <csv>.")
        return
    result = call_tool("plot_label_hist", csv_path=csv_path, label=label, bins=bins)
    if result.ok and isinstance(result.data, dict):
        path = result.data.get("image_path") if isinstance(result.data, dict) else None
        if path:
            print(f"Saved histogram to: {path}")
        else:
            print(result.to_json())
    else:
        try:
            print(json.dumps(result.to_json(), indent=2))
        except Exception:
            print(result.to_json())


def cmd_plot_feature_vs_label(ctx: CommandContext, user: str) -> None:
    parts = user.split()
    if len(parts) < 3:
        print("Usage: :plot_feature_vs_label <feature> <label>")
        return
    feature = parts[1]
    label = parts[2]
    csv_path = _csv_path_from_space(ctx.space)
    if not csv_path:
        print("No CSV path detected. Ensure you launched with --scope <csv>.")
        return
    result = call_tool("plot_feature_vs_label", csv_path=csv_path, feature=feature, label=label)
    if result.ok and isinstance(result.data, dict):
        path = result.data.get("image_path") if isinstance(result.data, dict) else None
        if path:
            print(f"Saved plot to: {path}")
        else:
            print(result.to_json())
    else:
        try:
            print(json.dumps(result.to_json(), indent=2))
        except Exception:
            print(result.to_json())


def cmd_corr_heatmap(ctx: CommandContext, user: str) -> None:
    parts = user.split(maxsplit=1)
    columns = None
    if len(parts) == 2 and parts[1].strip():
        raw = parts[1].strip()
        columns = [s.strip() for s in raw.split(',') if s.strip()]
    csv_path = _csv_path_from_space(ctx.space)
    if not csv_path:
        print("No CSV path detected. Ensure you launched with --scope <csv>.")
        return
    kwargs = {"csv_path": csv_path}
    if columns:
        kwargs["columns"] = columns
    result = call_tool("correlation_heatmap", **kwargs)
    if result.ok and isinstance(result.data, dict):
        path = result.data.get("image_path") if isinstance(result.data, dict) else None
        if path:
            print(f"Saved correlation heatmap to: {path}")
        else:
            print(result.to_json())
    else:
        try:
            print(json.dumps(result.to_json(), indent=2))
        except Exception:
            print(result.to_json())


def cmd_pareto(ctx: CommandContext, user: str) -> None:
    parts = user.split()
    if len(parts) < 3:
        print("Usage: :pareto <x_col> <y_col>")
        return
    x_col = parts[1]
    y_col = parts[2]
    csv_path = _csv_path_from_space(ctx.space)
    if not csv_path:
        print("No CSV path detected. Ensure you launched with --scope <csv>.")
        return
    result = call_tool("pareto_front", csv_path=csv_path, x_col=x_col, y_col=y_col)
    if result.ok and isinstance(result.data, dict):
        path = result.data.get("image_path") if isinstance(result.data, dict) else None
        if path:
            print(f"Saved Pareto plot to: {path}")
        else:
            print(result.to_json())
    else:
        try:
            print(json.dumps(result.to_json(), indent=2))
        except Exception:
            print(result.to_json())


def handle_repl_command(ctx: CommandContext, user: str) -> bool:
    cmd = user[1:].strip().lower()
    if cmd == "help":
        cmd_help(ctx, user)
        return True
    if cmd == "features":
        cmd_features(ctx, user)
        return True
    if cmd == "labels":
        cmd_labels(ctx, user)
        return True
    if cmd == "preview":
        cmd_preview(ctx, user)
        return True
    if cmd == "model":
        cmd_model(ctx, user)
        return True
    if cmd.startswith("describe"):
        cmd_describe(ctx, user)
        return True
    if cmd.startswith("best-preds"):
        cmd_best_preds(ctx, user)
        return True
    if cmd == "tools":
        cmd_tools(ctx, user)
        return True
    if cmd.startswith("call"):
        cmd_call(ctx, user)
        return True
    if cmd.startswith("plot_label_hist") or cmd.startswith("hist"):
        cmd_plot_label_hist(ctx, user)
        return True
    if cmd.startswith("plot_feature_vs_label"):
        cmd_plot_feature_vs_label(ctx, user)
        return True
    if cmd.startswith("corr_heatmap"):
        cmd_corr_heatmap(ctx, user)
        return True
    if cmd.startswith("pareto"):
        cmd_pareto(ctx, user)
        return True
    return False

