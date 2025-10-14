import textwrap
from typing import Optional, List
from .search_space import SearchSpace, summarize_search_space, preview_df

ASSISTANT_INSTRUCTIONS = """You are an EDBO assistant. Base answers on the provided search space context and retrieved repo snippets.
Tasks you support:
- List features
- List values for a feature (concise)
- List label/objective columns
- Show a preview (already provided – just reference)
- Provide concise Python snippets if explicitly asked (prefer using repo APIs)
- Clarify ambiguous queries briefly
Keep answers compact with short bullet lists when enumerating.
If a requested feature is unknown, state that and offer known features.
Grounded code policy:
- Only use APIs and classes that appear in the retrieved context or codebase summary; do not invent functions.
- Favor edbo.plus.optimizer_botorch.EDBOplus().run(...) for running rounds; avoid direct calls to internal helpers (e.g., build_and_optimize_model).
- If the needed API isn’t found in retrieved context, ask a brief clarification instead of guessing.
When the user asks for code, ensure the snippet is self-contained and complete (imports, object creation, minimal params) and keep it under ~30 lines. Do not cut off mid-sentence.

Tool use policy:
- Call a tool when the question requires precise numeric results from a CSV or computation that cannot be reliably inferred from text.
- If a suitable tool exists, respond with a single tool call (OpenAI function call) with JSON args; otherwise answer directly.
- Prefer at most one tool hop. After a tool returns, summarize the result succinctly and include provenance (e.g., which file/column).
""".strip()


def build_user_prompt(user_input: str, space: SearchSpace, include_preview: bool = False, feature_focus: Optional[str] = None, descriptor_db=None) -> str:
    summary = summarize_search_space(space)
    extra = []
    if feature_focus and feature_focus in space.feature_values:
        vals = space.feature_values[feature_focus]
        val_text = ", ".join(map(str, vals[:30])) + (" …" if len(vals) > 30 else "")
        extra.append(f"Values for {feature_focus}: {val_text}")
        if descriptor_db:
            desc = descriptor_db.get(feature_focus)
            if desc:
                extra.append(f"Definition of {feature_focus}: {desc}")
    if descriptor_db and not feature_focus:
        defs = []
        for f in space.features[:8]:
            d = descriptor_db.get(f)
            if d:
                defs.append(f"{f}: {d}")
        if defs:
            extra.append("Descriptor definitions (subset):\n- " + "\n- ".join(defs))
    if include_preview:
        extra.append("Search space preview (top 5 rows):\n" + preview_df(space.df, 5))
    context = f"Search Space Context\n{summary}\n\n" + ("\n".join(extra) if extra else "")
    prompt = textwrap.dedent(f"""
    {context}

    User question: {user_input}
    """)
    return prompt.strip()


def infer_feature_mention(user_text: str, features: List[str]) -> Optional[str]:
    q = user_text.lower()
    best = None
    for f in features:
        if f.lower() in q:
            best = f
            break
    return best
