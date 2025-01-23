from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass

import tiktoken
from aleph_alpha_client import AsyncClient, Client, CompletionRequest, Prompt

from .base import AsyncProvider, msg_as_str


@dataclass
class AlephAlphaProvider(AsyncProvider):
    MODEL_INFO = {
        "luminous-base": {"prompt": 6.6, "completion": 7.6, "token_limit": 2048},
        "luminous-extended": {"prompt": 9.9, "completion": 10.9, "token_limit": 2048},
        "luminous-supreme": {"prompt": 38.5, "completion": 42.5, "token_limit": 2048},
        "luminous-supreme-control": {
            "prompt": 48.5,
            "completion": 53.6,
            "token_limit": 2048,
        },
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

    def _from_dict(self, data: dict) -> CompletionRequest:
        return CompletionRequest(
            prompt=Prompt.from_text(data.pop("prompt")),
            **data,
        )

    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> dict:
        return {
            "prompt": prompt,
            "temperature": temperature,
            "maximum_tokens": kwargs.pop("maximum_tokens", max_tokens),
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        response = self.client.complete(request=self._from_dict(data), model=self.model)
        return {
            "completion": t.cast(str, response.completions[0].completion),
            "tokens_prompt": response.num_tokens_prompt_total,
            "tokens_completion": response.num_tokens_generated,
        }

    async def _acomplete(self, data: dict) -> dict:
        async with self.async_client as client:
            response = await client.complete(request=self._from_dict(data), model=self.model)
        return {
            "completion": t.cast(str, response.completions[0].completion),
            "tokens_prompt": response.num_tokens_prompt_total,
            "tokens_completion": response.num_tokens_generated,
        }
