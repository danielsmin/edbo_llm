from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatConfig:
    model_name: str = "gpt-4o-mini"
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    openai_api_key: Optional[str] = None
    # RAG config
    rag_enable: bool = True
    rag_top_k: int = 4
    provider: str = "openai"  # "openai" or "gemini"
    gemini_api_key: Optional[str] = None
