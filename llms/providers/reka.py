from __future__ import annotations

import typing as t
from dataclasses import dataclass

import tiktoken
from reka.client import AsyncReka, Reka

from .base import StreamProvider, msg_as_str


@dataclass
class RekaProvider(StreamProvider):
    MODEL_INFO = {
        "reka-edge": {"prompt": 0.4, "completion": 1.0, "token_limit": 128000},
        "reka-flash": {"prompt": 0.8, "completion": 2.0, "token_limit": 128000},
        "reka-core": {"prompt": 3.0, "completion": 15.0, "token_limit": 128000},
    }

    def __post_init__(self):
        self.model = self.model or "reka-core"
        self.client = Reka(api_key=self.api_key)
        self.async_client = AsyncReka(api_key=self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        # Reka uses the same tokenizer as OpenAI
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return sum([len(enc.encode(msg_as_str([message]))) for message in content])

    def _prepare_input(
        self,
        prompt: str,
        history: list[dict] | None = None,
        system_message: str | list[dict] | None = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> dict:
        messages = [{"content": prompt, "role": "user"}]

        if history:
            messages = [*history, *messages]

        if isinstance(system_message, str):
            messages = [{"role": "system", "content": system_message}, *messages]
        elif isinstance(system_message, list):
            messages = [*system_message, *messages]

        return {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        response = self.client.chat.create(model=self.model, **data)
        return {
            "completion": t.cast(str, response.responses[0].message.content),
            "tokens_prompt": response.usage.input_tokens,
            "tokens_completion": response.usage.output_tokens,
            "latency": self.latency,
        }

    async def _acomplete(self, data: dict) -> dict:
        response = await self.async_client.chat.create(model=self.model, **data)
        return {
            "completion": t.cast(str, response.responses[0].message.content),
            "tokens_prompt": response.usage.input_tokens,
            "tokens_completion": response.usage.output_tokens,
            "latency": self.latency,
        }

    def _complete_stream(self, data: dict) -> t.Iterator[str]:
        for r in self.client.chat.create_stream(model=self.model, **data):
            yield t.cast(str, r.responses[0].chunk.content)

    async def _acomplete_stream(self, data: dict) -> t.AsyncIterator[str]:
        async for chunk in self.async_client.chat.create_stream(model=self.model, **data):
            yield t.cast(str, chunk.responses[0].chunk.content)
