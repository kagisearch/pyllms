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
    }

    def __post_init__(self):
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)

    def count_tokens(self, content: str | dict | list) -> int:
        if isinstance(content, str):
            messages = t.cast(
                list,
                [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            )
        elif isinstance(content, dict):
            messages = t.cast(list, [content])
        else:
            messages = content

        return self.client.messages.count_tokens(
            model=self.model,
            messages=messages,
        ).input_tokens

    def _prepare_input(
        self,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: list[str] | None = None,
        ai_prompt: str = "",
        system_message: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        if history is None:
            history_prompt = ""
        else:
            history_text_list = []
            for message in history:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    role_prompt = anthropic.HUMAN_PROMPT
                elif role == "assistant":
                    role_prompt = anthropic.AI_PROMPT
                else:
                    msg = f"Invalid role {role}, role must be user or assistant."
                    raise ValueError(msg)

                formatted_message = f"{role_prompt}{content}"
                history_text_list.append(formatted_message)

            history_prompt = "".join(history_text_list)

        if system_message is None:
            system_prompts = ""
        else:
            if not self.model.startswith(("claude-2", "claude-3")):
                msg = "System message only available for Claude-2+ model"
                raise ValueError(msg)
            system_prompts = f"{system_message.rstrip()}"

        formatted_prompt = (
            f"{system_prompts}{history_prompt}{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}{ai_prompt}"
        )

        max_tokens_to_sample = kwargs.pop("max_tokens_to_sample", max_tokens)

        if stop_sequences is None:
            stop_sequences = [anthropic.HUMAN_PROMPT]
        return {
            "prompt": formatted_prompt,
            "temperature": temperature,
            "max_tokens_to_sample": max_tokens_to_sample,
            "stop_sequences": stop_sequences,
            "stream": stream,
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        response = self.client.messages.create(model=self.model, **data)
        return {
            "completion": response.content[0].text,
            "tokens_prompt": response.usage.input_tokens,
            "tokens_completion": response.usage.output_tokens,
        }

    async def _acomplete(self, data: dict) -> dict:
        response = await self.async_client.messages.create(model=self.model, **data)
        return {
            "completion": response.content[0].text,
            "tokens_prompt": response.usage.input_tokens,
            "tokens_completion": response.usage.output_tokens,
        }

    def _complete_stream(self, data: dict) -> t.Iterator[str]:
        with self.client.messages.stream(model=self.model, **data) as stream_manager:
            yield from stream_manager.text_stream

    async def _acomplete_stream(self, data: dict) -> t.AsyncIterator[str]:
        async with self.async_client.messages.stream(model=self.model, **data) as stream_manager:
            async for text in stream_manager.text_stream:
                yield text
