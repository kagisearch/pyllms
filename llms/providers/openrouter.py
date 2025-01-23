from __future__ import annotations

import typing as t
from dataclasses import dataclass

import tiktoken
from openai import AsyncOpenAI, OpenAI

from .base import StreamProvider, msg_as_str


@dataclass
class OpenRouterProvider(StreamProvider):
    MODEL_INFO = {
        "nvidia/llama-3.1-nemotron-70b-instruct": {
            "prompt": 0.35,
            "completion": 0.4,
            "token_limit": 131072,
            "is_chat": True,
        },
        "x-ai/grok-2": {
            "prompt": 5.0,
            "completion": 10.0,
            "token_limit": 32768,
            "is_chat": True,
        },
        "nousresearch/hermes-3-llama-3.1-405b:free": {
            "prompt": 0.0,
            "completion": 0.0,
            "token_limit": 8192,
            "is_chat": True,
        },
        "google/gemini-flash-1.5-exp": {
            "prompt": 0.0,
            "completion": 0.0,
            "token_limit": 1000000,
            "is_chat": True,
        },
        "liquid/lfm-40b": {
            "prompt": 0.0,
            "completion": 0.0,
            "token_limit": 32768,
            "is_chat": True,
        },
        "mistralai/ministral-8b": {
            "prompt": 0.1,
            "completion": 0.1,
            "token_limit": 128000,
            "is_chat": True,
        },
        "qwen/qwen-2.5-72b-instruct": {
            "prompt": 0.35,
            "completion": 0.4,
            "token_limit": 131072,
            "is_chat": True,
        },
        "x-ai/grok-2-1212": {
            "prompt": 2.0,
            "completion": 10.0,
            "token_limit": 131072,
            "is_chat": True,
        },
        "amazon/nova-pro-v1": {
            "prompt": 0.8,
            "completion": 3.2,
            "token_limit": 300000,
            "is_chat": True,
            "image_input": 1.2,
        },
        "qwen/qwq-32b-preview": {
            "prompt": 0.12,
            "completion": 0.18,
            "token_limit": 32768,
            "is_chat": True,
        },
        "mistralai/mistral-large-2411": {
            "prompt": 2.0,
            "completion": 6.0,
            "token_limit": 128000,
            "is_chat": True,
        },
    }

    def __post_init__(self):
        super().__post_init__()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    @property
    def is_chat_model(self) -> bool:
        return self.MODEL_INFO[self.model]["is_chat"]

    def _count_tokens(self, content: list[dict]) -> int:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        formatting_token_count = 4
        messages = content
        messages_text = [msg_as_str([message]) for message in messages]
        tokens = [enc.encode(t, disallowed_special=()) for t in messages_text]

        n_tokens_list = []
        for token, message in zip(tokens, messages):
            n_tokens = len(token) + formatting_token_count
            if "name" in message:
                n_tokens += -1
            n_tokens_list.append(n_tokens)
        return sum(n_tokens_list)

    def _prepare_input(
        self,
        prompt: str,
        history: list[dict] | None = None,
        system_message: str | list[dict] | None = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        messages = [{"role": "user", "content": prompt}]

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
            "extra_headers": {
                "HTTP-Referer": kwargs.get("site_url", ""),
                "X-Title": kwargs.get("app_name", ""),
            },
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        response = self.client.chat.completions.create(model=self.model, stream=False, **data)
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
        }

    async def _acomplete(self, data: dict) -> dict:
        response = await self.async_client.chat.completions.create(model=self.model, stream=False, **data)
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
        }

    def _complete_stream(self, data: dict) -> t.Iterator[str]:
        for chunk in self.client.chat.completions.create(model=self.model, stream=True, **data):
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    async def _acomplete_stream(self, data: dict) -> t.AsyncIterator[str]:
        async for chunk in await self.async_client.chat.completions.create(model=self.model, stream=True, **data):
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
