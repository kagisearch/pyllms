from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from openai import AsyncOpenAI, OpenAI

from .base import AsyncProvider


@dataclass
class DeepSeekProvider(AsyncProvider):
    MODEL_INFO = {
        "deepseek-chat": {
            "prompt": 0.14,
            "completion": 0.28,
            "token_limit": 128000,
            "is_chat": True,
            "output_limit": 8192,
        },
        "deepseek-coder": {
            "prompt": 0.14,
            "completion": 0.28,
            "token_limit": 128000,
            "is_chat": True,
            "output_limit": 8192,
        },
    }

    def __post_init__(self):
        super().__post_init__()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
        )

    @property
    def is_chat_model(self) -> bool:
        return self.MODEL_INFO[self.model]["is_chat"]

    def _count_tokens(self, content: list[dict]) -> int:
        # DeepSeek uses the same tokenizer as OpenAI
        enc = tiktoken.encoding_for_model(self.model)
        formatting_token_count = 4
        messages = content
        messages_text = ["".join(message.values()) for message in messages]
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
