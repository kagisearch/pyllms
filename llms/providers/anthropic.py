from __future__ import annotations

import typing as t
from dataclasses import dataclass

import anthropic

from .base import StreamProvider


@dataclass
class AnthropicProvider(StreamProvider):
    MODEL_INFO = {
        "claude-instant-v1.1": {
            "prompt": 1.63,
            "completion": 5.51,
            "token_limit": 9000,
        },
        "claude-instant-v1": {"prompt": 1.63, "completion": 5.51, "token_limit": 9000},
        "claude-v1": {"prompt": 11.02, "completion": 32.68, "token_limit": 9000},
        "claude-v1-100k": {
            "prompt": 11.02,
            "completion": 32.68,
            "token_limit": 100_000,
        },
        "claude-instant-1": {
            "prompt": 1.63,
            "completion": 5.51,
            "token_limit": 100_000,
        },
        "claude-instant-1.2": {
            "prompt": 1.63,
            "completion": 5.51,
            "token_limit": 100_000,
            "output_limit": 4_096,
        },
        "claude-2.1": {
            "prompt": 8.00,
            "completion": 24.00,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
        "claude-3-haiku-20240307": {
            "prompt": 0.25,
            "completion": 1.25,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
        "claude-3-sonnet-20240229": {
            "prompt": 3.00,
            "completion": 15,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
        "claude-3-opus-20240229": {
            "prompt": 15.00,
            "completion": 75,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
        "claude-3-5-sonnet-20240620": {
            "prompt": 3.00,
            "completion": 15,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
        "claude-3-5-sonnet-20241022": {
            "prompt": 3.00,
            "completion": 15,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
        "claude-3-5-sonnet-latest": {
            "prompt": 3.00,
            "completion": 15,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
    }

    def __post_init__(self):
        super().__post_init__()
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        return self.client.messages.count_tokens(
            model=self.model,
            messages=t.cast(t.Any, content),
        ).input_tokens

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.messages.create(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        return {
            "completion": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        response = await self.async_client.messages.create(
            model=self.model, messages=t.cast(t.Any, messages), **kwargs
        )
        return {
            "completion": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        with self.client.messages.stream(
            model=self.model, messages=t.cast(t.Any, messages), **kwargs
        ) as stream_manager:
            yield from stream_manager.text_stream

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        async with self.async_client.messages.stream(
            model=self.model, messages=t.cast(t.Any, messages), **kwargs
        ) as stream_manager:
            async for text in stream_manager.text_stream:
                yield text
