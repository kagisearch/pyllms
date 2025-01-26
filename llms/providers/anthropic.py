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

    @staticmethod
    def _prepare_messages(messages: list[dict]) -> tuple[list[dict], str | None]:
        system = next((m["content"] for m in reversed(messages) if m["role"] == "system"), None)
        if not system:
            return messages, None
        messages = [m for m in messages if m["role"] != "system"]
        return messages, system

    def complete(self, messages: list[dict], **kwargs) -> dict:
        messages, system = self._prepare_messages(messages)
        response = self.client.messages.create(
            model=self.model, messages=t.cast(t.Any, messages), system=system or anthropic.NOT_GIVEN, **kwargs
        )
        return {
            "completion": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        messages, system = self._prepare_messages(messages)
        response = await self.async_client.messages.create(
            model=self.model, messages=t.cast(t.Any, messages), system=system or anthropic.NOT_GIVEN, **kwargs
        )
        return {
            "completion": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        messages, system = self._prepare_messages(messages)
        with self.client.messages.stream(
            model=self.model, messages=t.cast(t.Any, messages), system=system or anthropic.NOT_GIVEN, **kwargs
        ) as stream_manager:
            yield from stream_manager.text_stream

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        messages, system = self._prepare_messages(messages)
        async with self.async_client.messages.stream(
            model=self.model, messages=t.cast(t.Any, messages), system=system or anthropic.NOT_GIVEN, **kwargs
        ) as stream_manager:
            async for text in stream_manager.text_stream:
                yield text
