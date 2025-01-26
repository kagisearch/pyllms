from __future__ import annotations

import typing as t
from dataclasses import dataclass

import anthropic

from .base import ModelInfo, StreamProvider


@dataclass
class AnthropicProvider(StreamProvider):
    MODEL_INFO = {
        "claude-instant-v1.1": ModelInfo(prompt_cost=1.63, completion_cost=5.51, context_limit=9000),
        "claude-instant-v1": ModelInfo(prompt_cost=1.63, completion_cost=5.51, context_limit=9000),
        "claude-v1": ModelInfo(prompt_cost=11.02, completion_cost=32.68, context_limit=9000),
        "claude-v1-100k": ModelInfo(prompt_cost=11.02, completion_cost=32.68, context_limit=100_000),
        "claude-instant-1": ModelInfo(prompt_cost=1.63, completion_cost=5.51, context_limit=100_000),
        "claude-instant-1.2": ModelInfo(
            prompt_cost=1.63, completion_cost=5.51, context_limit=100_000, output_limit=4_096
        ),
        "claude-2.1": ModelInfo(prompt_cost=8.00, completion_cost=24.00, context_limit=200_000, output_limit=4_096),
        "claude-3-haiku-20240307": ModelInfo(
            prompt_cost=0.25, completion_cost=1.25, context_limit=200_000, output_limit=4_096
        ),
        "claude-3-sonnet-20240229": ModelInfo(
            prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=4_096
        ),
        "claude-3-opus-20240229": ModelInfo(
            prompt_cost=15.00, completion_cost=75, context_limit=200_000, output_limit=4_096
        ),
        "claude-3-5-sonnet-20240620": ModelInfo(
            prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=4_096
        ),
        "claude-3-5-sonnet-20241022": ModelInfo(
            prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=4_096
        ),
        "claude-3-5-sonnet-latest": ModelInfo(
            prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=4_096
        ),
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
