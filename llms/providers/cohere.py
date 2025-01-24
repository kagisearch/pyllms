from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass

import cohere

from .base import StreamProvider, msg_as_str


@dataclass
class CohereProvider(StreamProvider):
    MODEL_INFO = {
        "command": {"prompt": 15.0, "completion": 15, "token_limit": 2048},
        "command-nightly": {
            "prompt": 15.0,
            "completion": 15,
            "token_limit": 4096,
        },
    }

    def __post_init__(self):
        super().__post_init__()
        api_key = self.api_key or os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(api_key)
        self.async_client = cohere.AsyncClient(api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        return len(self.client.tokenize(text=msg_as_str(content), model=self.model).tokens)

    def complete(self, messages: list[dict], **kwargs) -> dict:
        return {
            "completion": self.client.chat(
                model=self.model,
                message=messages[0]["content"] if len(messages) == 1 else msg_as_str(messages),
                **kwargs,
            ).text
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        async with self.async_client as client:
            return {
                "completion": (
                    await client.chat(
                        model=self.model,
                        message=messages[0]["content"] if len(messages) == 1 else msg_as_str(messages),
                        **kwargs,
                    )
                ).text
            }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        for token in self.client.chat_stream(
            model=self.model,
            message=messages[0]["content"] if len(messages) == 1 else msg_as_str(messages),
            **kwargs,
        ):
            yield t.cast(cohere.types.streamed_chat_response.TextGenerationStreamedChatResponse, token).text

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        async with self.async_client as client:
            async for r in client.chat_stream(
                model=self.model,
                message=messages[0]["content"] if len(messages) == 1 else msg_as_str(messages),
                **kwargs,
            ):
                yield t.cast(cohere.types.streamed_chat_response.TextGenerationStreamedChatResponse, r).text
