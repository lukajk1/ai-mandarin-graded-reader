"""A scripted OpenAI-compatible endpoint for tests."""

import json

import httpx
import pytest

from mandarin_core.llm import ChatClient


class FakeLLM:
    """Replies with `replies` in order and records every request body."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        text = self.replies[min(len(self.requests), len(self.replies)) - 1]
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    def client(self, **kwargs) -> ChatClient:
        return ChatClient(
            base_url="http://fake/v1",
            model="fake-model",
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )


@pytest.fixture
def make_llm():
    return FakeLLM
