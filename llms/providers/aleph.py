from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass

import tiktoken
from aleph_alpha_client import AsyncClient, Client, CompletionRequest, Prompt

from .base import AsyncProvider, ModelInfo, msg_as_str


@dataclass
class AlephAlphaProvider(AsyncProvider):
    MODEL_INFO = {
        "luminous-base": ModelInfo(prompt_cost=6.6, completion_cost=7.6, context_limit=2048),
        "luminous-extended": ModelInfo(prompt_cost=9.9, completion_cost=10.9, context_limit=2048),
        "luminous-supreme": ModelInfo(prompt_cost=38.5, completion_cost=42.5, context_limit=2048),
        "luminous-supreme-control": ModelInfo(prompt_cost=48.5, completion_cost=53.6, context_limit=2048),
    }

    def __post_init__(self):
        if not (host := os.getenv("ALEPHALPHA_HOST")):
            msg = "ALEPHALPHA_HOST environment variable is required"
            raise Exception(msg)
        self.client = Client(self.api_key, host)
        self.async_client = AsyncClient(self.api_key, host)

    def _count_tokens(self, content: list[dict]) -> int:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(msg_as_str(content)))

    @staticmethod
    def prepare_input(
        messages: list[dict],
        **kwargs,
    ) -> CompletionRequest:
        text = str(messages[0]["content"]) if len(messages) == 1 else msg_as_str(messages)
        if max_tokens := kwargs.pop("max_tokens", None):
            kwargs["maximum_tokens"] = max_tokens

        return CompletionRequest(
            prompt=Prompt.from_text(text),
            **kwargs,
        )

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.complete(request=self.prepare_input(messages, **kwargs), model=self.model)
        return {
            "completion": t.cast(str, response.completions[0].completion),
            "prompt_tokens": response.num_tokens_prompt_total,
            "completion_tokens": response.num_tokens_generated,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        async with self.async_client as client:
            response = await client.complete(request=self.prepare_input(messages, **kwargs), model=self.model)
        return {
            "completion": t.cast(str, response.completions[0].completion),
            "prompt_tokens": response.num_tokens_prompt_total,
            "completion_tokens": response.num_tokens_generated,
        }
