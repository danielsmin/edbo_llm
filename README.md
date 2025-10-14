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

## Removed / deprecated

- `code_summary.py` — No longer used; the minimal project root helper is inlined into `cli.py`.
- `llm.py` — Deprecated redundant entry point. Use `python -m edbo.llm` instead of `python -m edbo.llm.llm`.

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

Disable RAG if dependencies aren’t installed:

```bash
python -m edbo.llm ... --rag-disable
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
- `--rag-disable`: Disable local RAG (Chroma + MiniLM embeddings).
- `--rag-top-k`: How many retrieved chunks to include when RAG is enabled.

RAG indexing options (when enabled):
- `--rag-code-dirs`: Comma-separated directories to index (relative to project root). If omitted, the project root is used.
- `--rag-code-glob`: File pattern to index (default: `*.py`).
- `--rag-exclude`: Comma-separated substrings to skip (e.g., `__pycache__,.ipynb_checkpoints`).

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

## Add your own tools

Use the decorator (minimal):

```python
from edbo.llm.utils import register_tool

@register_tool(description="Short human description")
def my_tool(x: int, y: int) -> dict:
    """Optional docstring becomes docs for humans."""
    return {"sum": x + y}
```

Or bulk-register plain functions without decorators:

```python
# Example: we can add a helper like register_tools({"foo": foo, "bar": bar}) if you prefer.
# Ask us to wire it and the CLI will see them under :tools / :call as well.
```

## Tips

- To silence tokenizer fork warnings, the CLI sets `TOKENIZERS_PARALLELISM=false`.
- Keep CSV paths absolute to avoid ambiguity when calling tools.
- If RAG is not installed or desired, pass `--rag-disable`.
