"""EDBO LLM CLI

High-level flow:
1) Parse CLI flags (scope CSV, provider/model, RAG options).
2) Load scope into a SearchSpace (features/labels, preview helper).
3) Initialize an LLM backend (OpenAI or Gemini) from ChatConfig.
4) Optionally enable lightweight local RAG (Chroma + sentence-transformers).
5) Enter a simple REPL supporting handy commands (e.g., :features, :preview).

This file intentionally keeps runtime wiring and UI concerns here, while
model abstractions, RAG utilities, and prompt builders live in siblings.
"""

import argparse
import os
import json
from pathlib import Path

from .config import ChatConfig  # runtime settings for the chat session
from .backends import build_backend, GeminiBackend  # model adapters
from .search_space import SearchSpace, preview_df  # CSV loader + convenience preview
from .prompting import (
    ASSISTANT_INSTRUCTIONS,  # stable system prompt with guardrails
    build_user_prompt,       # formats the user request with context
    infer_feature_mention,   # detects a referenced feature, if any
)
def _find_project_root(start: Path) -> Path:
    p = Path(start).resolve()
    for _ in range(8):
        if (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return p
        p = p.parent
    return Path(start).resolve()
from .descriptors import load_descriptor_db
from .utils import list_tools as _list_tools, call_tool as _call_tool

try:
    from . import rag as ragmod
    RAG_AVAILABLE = True
except Exception:
    ragmod = None
    RAG_AVAILABLE = False


def _is_code_request(text: str) -> bool:
    """Heuristic: does the user appear to ask for code?

    Used to tweak generation limits (e.g., give Gemini more tokens for code).
    """
    t = text.lower()
    return any(w in t for w in ["code", "snippet", "example", "write code"]) and not t.startswith(":")


def run_cli(space: SearchSpace, cfg: ChatConfig):
    """Run an interactive chat loop over the provided search space and config.

    - "space" holds the features/labels and preview of the loaded CSV.
    - "cfg" selects the LLM provider/model and runtime generation settings.
    """
    backend = build_backend(cfg)  # pick OpenAI or Gemini backend
    print("EDBO Chatbot. Type 'exit' or 'quit' to leave. ':help' for commands.")
    print(f"Model: {cfg.model_name}")
    base_system = ASSISTANT_INSTRUCTIONS  # core behavior instructions
    try:
        project_root = _find_project_root(Path(__file__).parent)
        # Load descriptor definitions if present to enrich :describe and prompts
        descriptor_db = load_descriptor_db(project_root)
        if descriptor_db:
            print("Loaded descriptor definitions.")
        else:
            print("Descriptor definitions not found (edbo/DESCRIPTOR.md).")
    except Exception:
        descriptor_db = None

    # Optionally enable local retrieval-augmented generation (no network calls).
    client = None
    embedder = None
    if cfg.rag_enable and RAG_AVAILABLE and ragmod is not None:
        root = _find_project_root(Path(__file__).parent)
        persist_dir = root / ".chroma"  # on-disk vector cache
        client = ragmod.get_client(persist_dir)
        embedder = ragmod.Embedder()  # sentence-transformers/all-MiniLM-L6-v2
        print(f"RAG: enabled (persist at {persist_dir})")
        try:
            # Index a subset of repo code for helpful context
            added_code = ragmod.ingest_codebase(
                client, embedder, root, include_dirs=[root / "edbo" / "plus"]
            )
            print(f"RAG: indexed code chunks: {added_code}")
        except Exception:
            pass
        try:
            # Summaries of each CSV column (names, samples, basic stats)
            added_csv = ragmod.ingest_csv_columns(
                client,
                embedder,
                space.df,
                Path(space.df.attrs.get('source', 'dataset.csv')),
                descriptor_lookup=descriptor_db,
            )
            print(f"RAG: indexed CSV column docs: {added_csv}")
        except Exception:
            pass
        try:
            # Optional: index tutorial notebook cells
            nb_path = root / "tutorials" / "1_CLI_example.ipynb"
            added_nb = ragmod.ingest_ipynb(client, embedder, root, nb_path)
            print(f"RAG: indexed tutorial cells: {added_nb}")
        except Exception:
            pass
    else:
        if cfg.rag_enable:
            print("RAG: unavailable (install chromadb and sentence-transformers), proceeding without retrieval.")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if user.startswith(":"):
            cmd = user[1:].strip().lower()
            if cmd == "help":
                # Lightweight REPL commands for quick inspection
                print("Commands: :help, :features, :labels, :preview, :model, :describe <feature>, :tools, :call <tool> <json-kwargs>, :best-preds [csv_path]")
                continue
            if cmd == "features":
                print("Features:", ", ".join(space.features))
                continue
            if cmd == "labels":
                print("Labels:", ", ".join(space.labels) if space.labels else "(none specified)")
                continue
            if cmd == "preview":
                print(preview_df(space.df, 5))
                continue
            if cmd == "model":
                print("Current model:", cfg.model_name)
                continue
            if cmd.startswith("describe"):
                parts = user.split(maxsplit=1)
                if len(parts) == 2:
                    feat_name = parts[1].strip()
                    if descriptor_db:
                        d = descriptor_db.get(feat_name)
                        print(d if d else f"No descriptor definition for: {feat_name}")
                    else:
                        print("Descriptor DB not loaded.")
                else:
                    print("Usage: :describe <feature>")
                continue
            if cmd.startswith("best-preds"):
                # Convenience wrapper around the registered 'best_predictions' tool.
                # Usage:
                #   :best-preds                 -> uses current scope CSV if available
                #   :best-preds /path/to.csv    -> uses provided path
                parts = user.split(maxsplit=1)
                if len(parts) == 2:
                    csv_path = parts[1].strip()
                else:
                    csv_path = str(Path(space.df.attrs.get('source', '') or '').resolve()) if space and getattr(space, 'df', None) is not None else ''
                if not csv_path:
                    print("No CSV path detected. Provide one: :best-preds /path/to/predictions.csv")
                    continue
                result = _call_tool("best_predictions", csv_path=csv_path)
                if result.ok and isinstance(result.data, dict):
                    _pretty_print_best_predictions(result.data, df=space.df, label_cols=space.labels)
                else:
                    try:
                        print(json.dumps(result.to_json(), indent=2))
                    except Exception:
                        print(result.to_json())
                continue
            if cmd == "tools":
                tools = _list_tools()
                if not tools:
                    print("No tools registered.")
                else:
                    for t in tools:
                        name = t.get("name", "?")
                        desc = t.get("description", "")
                        print(f"- {name}: {desc}")
                continue
            if cmd.startswith("call"):
                parts = user.split(maxsplit=2)
                if len(parts) < 2:
                    print("Usage: :call <tool> <json-kwargs>")
                    continue
                tool_name = parts[1]
                kwargs = {}
                if len(parts) == 3:
                    json_arg = parts[2]
                    try:
                        kwargs = json.loads(json_arg)
                        if not isinstance(kwargs, dict):
                            print("Error: JSON kwargs must be an object, e.g., {\"csv_path\": \"/path/file.csv\"}")
                            continue
                    except Exception as e:
                        print(f"JSON parse error: {e}")
                        continue
                result = _call_tool(tool_name, **kwargs)
                # Pretty-print known tools
                if tool_name == "best_predictions" and result.ok and isinstance(result.data, dict):
                    df_for_print = None
                    try:
                        src = space.df.attrs.get('source') if getattr(space, 'df', None) is not None else None
                        csv_kw = kwargs.get('csv_path') if isinstance(kwargs, dict) else None
                        if src and csv_kw and Path(src).resolve() == Path(csv_kw).resolve():
                            df_for_print = space.df
                    except Exception:
                        df_for_print = None
                    _pretty_print_best_predictions(result.data, df=df_for_print, label_cols=(space.labels if df_for_print is not None else None))
                else:
                    try:
                        print(json.dumps(result.to_json(), indent=2))
                    except Exception:
                        print(result.to_json())
                continue
        # Build the user/task context
        feat = infer_feature_mention(user, space.features)
        include_preview = any(w in user.lower() for w in ["preview", "search space", "table", "rows", "combinations"])
        ul = user.lower()
        rag_block = ""
        if cfg.rag_enable and RAG_AVAILABLE and ragmod is not None and client is not None and embedder is not None:
            try:
                # Fetch a few relevant snippets (extra for code requests)
                k = cfg.rag_top_k + 6 if "code" in ul else cfg.rag_top_k
                hits = ragmod.query(client, embedder, user, top_k=k)
                rag_block = ragmod.format_hits_for_prompt(hits) if hits else ""
            except Exception:
                pass
        # Merge retrieved snippets (if any) and build the final prompts
        augmented_system = base_system + ("\n\nRetrieved context:\n" + rag_block if rag_block else "")
        user_prompt = build_user_prompt(
            user,
            space,
            include_preview=include_preview,
            feature_focus=feat,
            descriptor_db=descriptor_db,
        )
        if any(w in ul for w in ["code", "snippet", "example"]) and isinstance(backend, GeminiBackend):
            # Give Gemini more room to produce code when asked
            backend.cfg.max_new_tokens = max(backend.cfg.max_new_tokens, 768)
        reply = backend.generate(augmented_system, user_prompt)
        print(f"Bot: {reply}")


def parse_args():
    """Define and parse CLI flags.

    Key flags:
    - --scope: CSV path combining features (+ optional labels).
    - --provider/--model: choose backend and model name.
    - --rag-disable/--rag-top-k: control local retrieval.
    - --openai-api-key/--gemini-api-key: override env vars.
    """
    p = argparse.ArgumentParser(description="EDBO Chatbot (OpenAI + local RAG cache)")
    p.add_argument("--scope", type=str, required=True, help="Path to scope CSV (features + optional label columns)")
    p.add_argument("--label-cols", type=str, default="", help="Comma-separated label columns present in CSV")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name (OpenAI) or Gemini model")
    p.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API key (optional, else env OPENAI_API_KEY)")
    p.add_argument("--provider", type=str, choices=["openai", "gemini"], default="openai", help="LLM provider")
    p.add_argument("--gemini-api-key", type=str, default=None, help="Gemini API key (optional, else env GEMINI_API_KEY)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--rag-disable", action="store_true", help="Disable local RAG cache and retrieval")
    p.add_argument("--rag-top-k", type=int, default=4, help="Top-k retrieved chunks to include from RAG")
    return p.parse_args()


def main():
    """Entry point used by __main__.py and by CLI wrappers.

    Creates the SearchSpace from the CSV path and wires ChatConfig from flags,
    then invokes the interactive REPL via run_cli().
    """
    args = parse_args()
    label_cols = [x.strip() for x in args.label_cols.split(',') if x.strip()]
    space = SearchSpace.from_csv(args.scope, label_cols)
    cfg = ChatConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        openai_api_key=args.openai_api_key,
        rag_enable=(not args.rag_disable),
        rag_top_k=max(0, int(args.rag_top_k)),
        provider=args.provider,
        gemini_api_key=args.gemini_api_key,
    )
    run_cli(space, cfg)

def _pretty_print_best_predictions(data: dict, df=None, label_cols=None) -> None:
    """Print a human-readable summary for the best_predictions tool output.

    If a DataFrame is provided, also render reaction conditions (all columns
    excluding prediction/label columns) for the selected rows.
    """
    def _fmt_value(v):
        try:
            # Format numbers nicely, otherwise fallback to str
            if isinstance(v, (int, float)):
                return f"{v:.4f}" if not isinstance(v, bool) else str(v)
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
            # Access row by label index if present else positional
            if idx in _df.index:
                row = _df.loc[idx]
            else:
                row = _df.iloc[int(idx)]
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
        # Print without raw row number; show conditions when possible
        print(f"- {title}: {val} (column: {col})")
        conds = _fmt_row_conditions(df, idx)
        if conds:
            print(f"  conditions: {conds}")
