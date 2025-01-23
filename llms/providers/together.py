from __future__ import annotations

import typing as t
from dataclasses import dataclass

import tiktoken
import together
from together import AsyncTogether, Together

from .base import StreamProvider, msg_as_str


@dataclass
class TogetherProvider(StreamProvider):
    MODEL_INFO = {
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {
            "prompt": 5.0,
            "completion": 5.0,
            "token_limit": 4096,
        },
    }

    def __post_init__(self):
        super().__post_init__()
        self.client = Together(api_key=self.api_key)
        self.async_client = AsyncTogether(api_key=self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        # Together uses the same tokenizer as OpenAI
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return sum([len(enc.encode(msg_as_str([message]))) for message in content])

    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        history: list[dict] | None = None,
        system_message: str | list[dict] | None = None,
        stream: bool = False,
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
            "stream": stream,
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        response = t.cast(
            together.types.ChatCompletionResponse,
            self.client.chat.completions.create(model=self.model, stream=False, **data),
        )
        assert response.choices
        assert response.choices[0].message
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
        }

    async def _acomplete(self, data: dict) -> dict:
        response = t.cast(
            together.types.ChatCompletionResponse,
            await self.async_client.chat.completions.create(model=self.model, **data),
        )
        assert response.choices
        assert response.choices[0].message
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
        }

    def _complete_stream(self, data: dict) -> t.Iterator[str]:
        for chunk in self.client.chat.completions.create(model=self.model, stream=True, **data):
            chunk = t.cast(together.types.ChatCompletionChunk, chunk)
            assert chunk.choices
            assert chunk.choices[0].delta
            s = chunk.choices[0].delta.content
            assert s
            yield s

    async def _acomplete_stream(self, data: dict) -> t.AsyncIterator[str]:
        async for chunk in t.cast(
            t.AsyncGenerator[together.types.ChatCompletionChunk, None],
            self.async_client.chat.completions.create(model=self.model, stream=True, **data),
        ):
            chunk = t.cast(together.types.ChatCompletionChunk, chunk)
            assert chunk.choices
            assert chunk.choices[0].delta
            s = chunk.choices[0].delta.content
            assert s
            yield s
