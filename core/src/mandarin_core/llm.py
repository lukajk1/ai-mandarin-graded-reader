"""Minimal client for OpenAI-compatible chat endpoints (llama-server, hosted providers)."""

import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


@dataclass(frozen=True)
class Completion:
    text: str
    latency_s: float
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def clean_output(text: str) -> str:
    """Strip reasoning blocks and code fences some models wrap around the passage."""
    text = _THINK_RE.sub("", text).strip()
    return _FENCE_RE.sub("", text).strip()


@dataclass
class ChatClient:
    base_url: str  # e.g. http://127.0.0.1:8080/v1
    model: str
    api_key: str | None = None
    timeout_s: float = 300.0
    # Merged into every request body, e.g. {"chat_template_kwargs": {"enable_thinking": False}}.
    extra_body: dict[str, Any] = field(default_factory=dict)
    transport: httpx.BaseTransport | None = None

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 600,
    ) -> Completion:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.extra_body,
        }
        start = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s, transport=self.transport) as http:
            response = http.post(
                f"{self.base_url.rstrip('/')}/chat/completions", json=body, headers=headers
            )
            response.raise_for_status()
        latency = time.perf_counter() - start
        data = response.json()
        usage = data.get("usage") or {}
        return Completion(
            text=clean_output(data["choices"][0]["message"]["content"] or ""),
            latency_s=latency,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
