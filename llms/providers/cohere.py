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

    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        return {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        return {
            "completion": self.client.chat(
                model=self.model,
                **data,
            ).text
        }

    async def _acomplete(self, data: dict) -> dict:
        async with self.async_client as client:
            return {
                "completion": (
                    await client.chat(
                        model=self.model,
                        **data,
                    )
                ).text
            }

    def _complete_stream(self, data: dict) -> t.Iterator[str]:
        for token in self.client.chat_stream(
            model=self.model,
            **data,
        ):
            yield t.cast(cohere.types.streamed_chat_response.TextGenerationStreamedChatResponse, token).text

    async def _acomplete_stream(self, data: dict) -> t.AsyncIterator[str]:
        async with self.async_client as client:
            async for r in client.chat_stream(
                model=self.model,
                **data,
            ):
                yield t.cast(cohere.types.streamed_chat_response.TextGenerationStreamedChatResponse, r).text
