import os
from typing import List

from .config import ChatConfig

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


class BaseBackend:
    def generate(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class OpenAIBackend(BaseBackend):
    def __init__(self, cfg: ChatConfig):
        if OpenAI is None:
            raise RuntimeError("openai package not installed. pip install openai")
        self.cfg = cfg
        key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY env var or pass --openai-api-key")
        os.environ.setdefault("OPENAI_API_KEY", key)
        self.client = OpenAI()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model_name,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_new_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover
            return f"[OpenAI error: {e}]"


class GeminiBackend(BaseBackend):
    def __init__(self, cfg: ChatConfig):
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai") from e
        key = cfg.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY env var or pass --gemini-api-key")
        genai.configure(api_key=key)
        self.cfg = cfg
        self._genai = genai

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            # Use the provided model name as-is; let the API surface invalid names.
            model = self._genai.GenerativeModel(
                self.cfg.model_name,
                system_instruction=system_prompt,
            )
            resp = model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": self.cfg.temperature,
                    "max_output_tokens": self.cfg.max_new_tokens,
                    "top_p": self.cfg.top_p,
                    "response_mime_type": "text/plain",
                },
            )
            # Primary: use resp.text when available
            if getattr(resp, "text", None):
                return (resp.text or "").strip()
            # Fallback: join any text parts from the first candidate
            try:
                cands = getattr(resp, "candidates", []) or []
                for cand in cands:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    parts = getattr(content, "parts", []) or []
                    texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                    if texts:
                        return "\n".join(texts).strip()
            except Exception:
                pass
            return "[Gemini produced no text]"
        except Exception as e:  # pragma: no cover
            return f"[Gemini error: {e}]"


def build_backend(cfg: ChatConfig) -> BaseBackend:
    if (cfg.provider or "openai").lower() == "gemini":
        return GeminiBackend(cfg)
    return OpenAIBackend(cfg)
