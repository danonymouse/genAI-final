"""Minimal client for any OpenAI-compatible chat endpoint.

One client covers every backend we might use, all free to run:
  * Ollama on Colab or a laptop     LLM_BASE_URL=http://localhost:11434/v1       (no key)
  * vLLM / llama.cpp server         LLM_BASE_URL=http://<host>:8000/v1           (no key)
  * hosted free tiers (e.g. Gemini, Groq) through their OpenAI-compatible URLs (key)

Open models are less reliable at strict JSON than hosted APIs, so the parser is lenient:
it strips <think>...</think> blocks and ``` fences and takes the outermost {...} object.
"""
from __future__ import annotations

import json
import re

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Pull the first complete JSON object out of a model reply."""
    text = _THINK_RE.sub("", text or "")
    text = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise LLMError(f"no JSON object in model reply: {text[:200]!r}")


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self._json_mode = True  # turned off for the session if the server rejects it

    def complete_json(self, system: str, user: str, max_tokens: int = 2000) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        r = httpx.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
        if r.status_code == 400 and self._json_mode:  # server doesn't support JSON mode
            self._json_mode = False
            body.pop("response_format")
            r = httpx.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            content = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected response shape: {r.text[:200]}") from e
        return extract_json(content)
