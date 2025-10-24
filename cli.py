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
from pathlib import Path

from .config import ChatConfig  # runtime settings for the chat session
from .backends import build_backend, GeminiBackend  # model adapters
from .search_space import SearchSpace  # CSV loader
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
from .utils import CommandContext, handle_repl_command

try:
    from . import rag as ragmod
    RAG_AVAILABLE = True
except Exception:
    ragmod = None
    RAG_AVAILABLE = False


#
# Note: Code-request heuristics are handled inline in run_cli for simplicity.


def run_cli(space: SearchSpace, cfg: ChatConfig):
    """Run an interactive chat loop over the provided search space and config."""
    # Permanently suppress HuggingFace tokenizers parallelism fork warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
        try:
            client = ragmod.get_client(persist_dir)
            embedder = ragmod.Embedder()  # sentence-transformers/all-MiniLM-L6-v2
            print(f"RAG: enabled (persist at {persist_dir})")
            try:
                added_code = ragmod.ingest_codebase(
                    client, embedder, root, include_dirs=[root / "edbo" / "plus"]
                )
                print(f"RAG: indexed code chunks: {added_code}")
            except Exception:
                pass
            try:
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
                nb_path = root / "tutorials" / "1_CLI_example.ipynb"
                added_nb = ragmod.ingest_ipynb(client, embedder, root, nb_path)
                print(f"RAG: indexed tutorial cells: {added_nb}")
            except Exception:
                pass
        except Exception:
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
            # Delegate REPL commands to command_utils
            ctx = CommandContext(space=space, cfg=cfg, descriptor_db=descriptor_db)
            if handle_repl_command(ctx, user):
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
        rag_enable=True,
        rag_top_k=max(0, int(args.rag_top_k)),
        provider=args.provider,
        gemini_api_key=args.gemini_api_key,
    )
    run_cli(space, cfg)

def _pretty_print_best_predictions(*args, **kwargs):  # Backward shim if imported elsewhere
    from .utils import _pretty_print_best_predictions as _pp
    return _pp(*args, **kwargs)
