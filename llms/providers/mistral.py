from __future__ import annotations

import typing as t
from dataclasses import dataclass

import tiktoken
from mistralai import Mistral

from .base import StreamProvider, from_raw, msg_as_str


@dataclass
class MistralProvider(StreamProvider):
    MODEL_INFO = {
        "mistral-tiny": {"prompt": 0.25, "completion": 0.25, "token_limit": 32_000},
        # new endpoint for mistral-tiny, mistral-tiny will be deprecated in ~June 2024
        "open-mistral-7b": {"prompt": 0.25, "completion": 0.25, "token_limit": 32_000},
        "mistral-small": {"prompt": 0.7, "completion": 0.7, "token_limit": 32_000},
        # new endpoint for mistral-small, mistral-small will be deprecated in ~June 2024
        "open-mixtral-8x7b": {"prompt": 0.7, "completion": 0.7, "token_limit": 32_000},
        "mistral-small-latest": {
            "prompt": 2.0,
            "completion": 6.0,
            "token_limit": 32_000,
        },
        "mistral-medium-latest": {
            "prompt": 2.7,
            "completion": 8.1,
            "token_limit": 32_000,
        },
        "mistral-large-latest": {
            "prompt": 3.0,
            "completion": 9.0,
            "token_limit": 32_000,
        },
        "open-mistral-nemo": {"prompt": 0.3, "completion": 0.3, "token_limit": 32_000},
    }

    def __post_init__(self):
        super().__post_init__()
        self.client = Mistral(api_key=self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        # TODO: update after Mistrar support count token in their SDK
        # use gpt 3.5 turbo for estimation now
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        formatting_token_count = 4
        messages = content
        messages_text = [msg_as_str([message]) for message in messages]
        tokens = [enc.encode(t, disallowed_special=()) for t in messages_text]

        n_tokens_list = []
        for token in tokens:
            n_tokens = len(token) + formatting_token_count
            n_tokens_list.append(n_tokens)
        return sum(n_tokens_list)

    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        history: list[dict] | None = None,
        stop_sequences: list[str] | None = None,
        system_message: str | None = None,
        safe_prompt: bool = False,
        random_seed: int | None = None,
        **kwargs,
    ) -> dict:
        if stop_sequences:
            msg = "Parameter `stop` is not supported"
            raise ValueError(msg)

        messages = history or []
        messages.extend(from_raw(prompt))
        if system_message:
            messages.extend(from_raw(system_message, "system"))

        return {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "safe_prompt": safe_prompt,
            "random_seed": random_seed,
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        with self.client as client:
            response = client.chat.complete(model=self.model, **data)
        assert response.choices
        return {
            "completion": response.choices[0].message.content,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
        }

    async def _acomplete(self, data: dict) -> dict:
        async with self.client as client:
            response = await client.chat.complete_async(model=self.model, **data)
        assert response.choices
        return {
            "completion": response.choices[0].message.content,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
        }

    def _complete_stream(self, data: dict) -> t.Iterator[str]:
        with self.client as client, client.chat.stream(model=self.model, **data) as stream:
            for chunk in stream:
                assert chunk.data.choices
                if c := chunk.data.choices[0].delta.content:
                    yield t.cast(str, c)

    async def _acomplete_stream(self, data: dict) -> t.AsyncIterator[str]:
        async with self.client as client:
            async for chunk in await client.chat.stream_async(model=self.model, **data):
                assert chunk.data.choices
                if c := chunk.data.choices[0].delta.content:
                    yield t.cast(str, c)
