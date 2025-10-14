import os
import json
from typing import List, Dict, Any

from .config import ChatConfig
from .utils import list_tools as _list_tools, call_tool as _call_tool

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

    def _build_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert registered tools into OpenAI tool specs."""
        tools: List[Dict[str, Any]] = []
        for t in _list_tools():
            name = t.get("name")
            if not name:
                continue
            desc = t.get("description") or ""
            params = t.get("input_schema") or {"type": "object", "properties": {}}
            # Ensure schema is a mapping; if not, wrap as empty object
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}}
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params,
                },
            })
        return tools

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            tools = self._build_openai_tools()
            resp = self.client.chat.completions.create(
                model=self.cfg.model_name,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_new_tokens,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else "none",
            )
            choice = resp.choices[0]
            msg = choice.message
            # If model requested a tool call, execute at most one and provide the result
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                tc = tool_calls[0]
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
                arg_str = getattr(fn, "arguments", "{}") if fn else "{}"
                kwargs: Dict[str, Any]
                try:
                    kwargs = json.loads(arg_str) if arg_str else {}
                    if not isinstance(kwargs, dict):
                        kwargs = {}
                except Exception:
                    kwargs = {}
                # Log the tool call for user visibility
                try:
                    pretty_args = json.dumps(kwargs)
                except Exception:
                    pretty_args = str(kwargs)
                print(f"[Tool call] {name} {pretty_args}")
                result = _call_tool(name, **kwargs) if name else None
                tool_content = json.dumps((result.to_json() if result else {"ok": False, "error": "invalid tool"}))
                # Append assistant msg (with tool_calls) and the tool result
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": getattr(tc, "id", ""), "content": tool_content})
                # One follow-up to get final answer
                resp2 = self.client.chat.completions.create(
                    model=self.cfg.model_name,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_new_tokens,
                    messages=messages,
                )
                return (resp2.choices[0].message.content or "").strip()
            # No tool call; just return content
            return (msg.content or "").strip()
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
