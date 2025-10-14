# edbo/llm

This package provides a lightweight, code-first CLI for chatting with an LLM over an experimental design search space (scope CSV), with optional local RAG.

## Files

- `__init__.py` — Marks this directory as a Python package.
- `__main__.py` — Module entry point; enables `python -m edbo.llm`.
- `cli.py` — The CLI and REPL. Parses flags, loads the scope CSV, wires the model backend, optionally enables local RAG, and handles chat commands (`:help`, `:features`, `:labels`, `:preview`, `:model`, `:describe`).
- `backends.py` — Model adapters for OpenAI and Gemini, exposing a common `generate(system, user)` API. Selected via `build_backend(ChatConfig)`.
- `config.py` — `ChatConfig` dataclass with runtime settings (model, provider, tokens/temperature/top_p, API keys, RAG flags).
- `prompting.py` — System instructions and prompt builders (formats user requests, infers feature mentions, builds previews).
- `search_space.py` — Loads the scope CSV into a DataFrame, tracks feature/label columns, and prints small previews.
- `rag.py` — Local RAG using ChromaDB + sentence-transformers. Can index repo code, notebook cells, and per-column CSV docs; provides simple query/format helpers.
- `descriptors.py` — Loads descriptor definitions (if present) to enrich `:describe <feature>` and augment prompts.

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
