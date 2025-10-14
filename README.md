# edbo/llm

A lightweight, code-first CLI for chatting with an LLM over an experimental design search space (scope CSV), with optional local RAG and a pluggable tool registry the assistant can call.

## Files

- `__init__.py` — Marks this directory as a Python package.
- `__main__.py` — Module entry point; enables `python -m edbo.llm`.
- `cli.py` — The CLI and REPL. Parses flags, loads the scope CSV, wires the model backend, optionally enables local RAG, and handles chat commands (`:help`, `:features`, `:labels`, `:preview`, `:model`, `:describe`, `:tools`, `:call`, `:best-preds`).
- `backends.py` — Model adapters for OpenAI and Gemini, exposing a common `generate(system, user)` API. Selected via `build_backend(ChatConfig)`.
- `config.py` — `ChatConfig` dataclass with runtime settings (model, provider, tokens/temperature/top_p, API keys, RAG flags).
- `prompting.py` — System instructions and prompt builders (formats user requests, infers feature mentions, builds previews).
- `search_space.py` — Loads the scope CSV into a DataFrame, tracks feature/label columns, and prints small previews.
- `rag.py` — Local RAG using ChromaDB + sentence-transformers. Can index repo code, notebook cells, and per-column CSV docs; provides simple query/format helpers.
- `descriptors.py` — Loads descriptor definitions (if present) to enrich `:describe <feature>` and augment prompts.
- `utils.py` — Tool registry: `@register_tool` decorator, `list_tools`, `call_tool`, JSON-safe result wrapper (`ToolResult`). Includes example tools.

## Run

OpenAI example:

```bash
export OPENAI_API_KEY="sk-..."
python -m edbo.llm \
  --scope /path/to/experiments_yield_and_cost.csv \
  --label-cols yield,cost \
  --provider openai \
  --model gpt-4o-mini
```

Gemini example:

```bash
export GEMINI_API_KEY="..."
python -m edbo.llm \
  --scope /path/to/experiments_yield_and_cost.csv \
  --label-cols yield,cost \
  --provider gemini \
  --model gemini-1.5-flash
```

## CLI flags (selected)

- `--scope`: Path to CSV defining your search space (or predictions table).
- `--label-cols`: Comma-separated labels/objectives present in the CSV (e.g., `yield,ee`).
- `--provider`: `openai` (default) or `gemini`.
- `--model`: Model name (e.g., `gpt-4o`, `gpt-4o-mini`, `gemini-1.5-flash`).
  

## REPL commands

- `:help` — Show commands.
- `:features` — List feature columns.
- `:labels` — List label/objective columns.
- `:preview` — Show a small preview of the current CSV.
- `:model` — Show the current model.
- `:describe <feature>` — Show descriptor definition if available.
- `:tools` — List registered tools from `utils.py`.
- `:call <tool> <json-kwargs>` — Invoke a tool by name with JSON args.
- `:best-preds [csv_path]` — Convenience wrapper for predictions summary (uses current scope when path is omitted).
 - `:plot_label_hist <label> [bins]` — Plot and save a histogram for a numeric label column.
 - `:hist <label> [bins]` — Short alias for `:plot_label_hist`.
 - `:plot_feature_vs_label <feature> <label>` — Plot feature vs label (scatter for numeric feature, boxplot for categorical).
 - `:corr_heatmap [col1,col2,...]` — Plot correlation heatmap for numeric columns (optionally restrict to a subset).
 - `:pareto <x_col> <y_col>` — Plot Pareto front for two objectives (assumes both are maximized). Note: this REPL command maps to the tool named `pareto_front`.

REPL vs tool names: REPL commands are prefixed with `:` (e.g., `:pareto`), while the underlying tool names used by auto-tool calls or `:call` can differ slightly (e.g., `pareto_front`).

## Auto tool use (OpenAI)

When using OpenAI, the backend exposes registered tools to the model and enables function-calling (`tool_choice="auto"`). The system prompt guides when to call a tool (for precise numeric answers from CSV/computation) vs answer directly. If the model requests a tool call, the CLI executes it once, returns the result to the model, and prints the final response. A small log line shows tool invocations, e.g.:

```
[Tool call] best_predictions {"csv_path": "/path/to/predictions.csv"}
```

Note: Auto tool use can slightly increase token usage due to tool schemas and an extra round-trip when a tool is invoked.

## Available tools (from `utils.py`)

- `echo(text: str) -> {"text": str}`
  - Description: Echo back the provided text (template/example tool).

- `best_predictions(csv_path: str) -> { best_predicted_mean, best_predicted_variance, best_expected_improvement }`
  - Description: Scan a predictions CSV and select overall best entries for:
    - max `*_predicted_mean`
    - max `*_predicted_variance`
    - max `*_expected_improvement`
  - Returns each as an object with `{column, value, row_index}`.

- `feature_values(csv_path: str, column: str, top_n: int = 50) -> {column, distinct, top[]}`
  - Description: Show the most frequent values of a column (with counts) and the number of distinct values.

- `label_stats(csv_path: str, label: str) -> {count, mean, std, min, median, max, quantiles}`
  - Description: Summary statistics for a numeric label column.

- `plot_label_hist(csv_path: str, label: str, bins: int = 30) -> {image_path, label}`
  - Description: Histogram of a numeric label; saves a PNG and returns its path.

- `plot_feature_vs_label(csv_path: str, feature: str, label: str) -> {image_path, feature, label}`
  - Description: Scatter if feature is numeric; otherwise a boxplot per category; saves PNG and returns its path.

- `correlation_heatmap(csv_path: str, columns: list[str] | None = None, top_k_pairs: int = 10) -> {image_path, top_pairs[]}`
  - Description: Correlation heatmap for numeric columns (or a provided subset). Also returns the top correlated pairs by absolute value.

- `pareto_front(csv_path: str, x_col: str, y_col: str) -> {image_path, pareto_indices[]}`
  - Description: Compute and plot the Pareto front assuming both objectives are maximized; returns plot path and row indices on the front.

- `top_n_by_metric(csv_path: str, metric_column: str, n: int = 10, prefer: "max"|"min" = "max") -> {metric, prefer, rows[]}`
  - Description: Select the top-N rows by a metric column (e.g., an acquisition value), returning row indices and values.

- `group_mean_label_by_feature(csv_path: str, feature: str, label: str) -> {feature, label, groups[]}`
  - Description: Group by a categorical feature and report count and mean of the label per category (sorted by mean desc).

- `describe_row(csv_path: str, row_index: int, exclude_suffixes?: list[str], exclude_columns?: list[str]) -> {row_index, conditions{}}`
  - Description: Display reaction conditions/inputs for one row, excluding prediction columns by default.

- `compare_rows(csv_path: str, row_a: int, row_b: int) -> {row_a, row_b, differences[]}`
  - Description: Compare two rows and list columns with differing values (prediction columns excluded by default).

Note on plots: plotting tools write PNGs to `.edbo_llm_artifacts/` and return the `image_path` so you can open them quickly.

## Add your own tools

Use the decorator:

```python
from edbo.llm.utils import register_tool

@register_tool(description="Short human description")
def my_tool(x: int, y: int) -> dict:
    """Optional docstring becomes docs for humans."""
    return {"sum": x + y}
```

## Tips

- To silence tokenizer fork warnings, the CLI sets `TOKENIZERS_PARALLELISM=false`.
- Keep CSV paths absolute to avoid ambiguity when calling tools.
- Plotting artifacts are saved under `.edbo_llm_artifacts/` in your current working directory.
