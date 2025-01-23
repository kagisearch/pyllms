from __future__ import annotations

import math
import os
from dataclasses import dataclass

import google.generativeai as genai

from .base import SyncProvider, msg_as_str


@dataclass
class GoogleGenAIProvider(SyncProvider):
    # cost is per million tokens
    MODEL_INFO = {
        # no support for "textembedding-gecko"
        "chat-bison-genai": {
            "prompt": 0.5,
            "completion": 0.5,
            "token_limit": 0,
            "uses_characters": True,
        },
        "text-bison-genai": {
            "prompt": 1.0,
            "completion": 1.0,
            "token_limit": 0,
            "uses_characters": True,
        },
        "gemini-1.5-pro": {
            "prompt": 3.5,
            "completion": 10.5,
            "token_limit": 128000,
            "uses_characters": True,
        },
        "gemini-1.5-pro-latest": {
            "prompt": 3.5,
            "completion": 10.5,
            "token_limit": 128000,
            "uses_characters": True,
        },
        "gemini-1.5-flash": {
            "prompt": 0.075,
            "completion": 0.3,
            "token_limit": 128000,
            "uses_characters": True,
        },
        "gemini-1.5-flash-latest": {
            "prompt": 0.075,
            "completion": 0.3,
            "token_limit": 128000,
            "uses_characters": True,
        },
        "gemini-1.5-pro-exp-0801": {
            "prompt": 3.5,
            "completion": 10.5,
            "token_limit": 128000,
            "uses_characters": True,
        },
    }

    def __post_init__(self):
        super().__post_init__()
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")

        self.client = genai.configure(api_key=api_key)  # type: ignore

        model = self.model
        if model.startswith("text-"):
            self.client = genai.generate_text  # type: ignore[private-import]
            self.mode = "text"
        else:
            self.client = genai.GenerativeModel(model)  # type: ignore[private-import]
            self.mode = "chat"

    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ) -> dict:
        temperature = max(temperature, 0.01)
        max_output_tokens = kwargs.pop("max_output_tokens", max_tokens)
        if self.mode == "chat":
            messages = kwargs.pop("messages", [])
            messages = messages + [prompt]
            model_inputs = {
                # "messages": messages,
                # "temperature": temperature,
                **kwargs,
            }
        else:
            model_inputs = {
                "prompt": prompt,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                **kwargs,
            }
        return model_inputs

    def _count_tokens(self, content: list[dict]) -> int:
        return self.client.count_tokens(msg_as_str(content)).total_tokens  # type: ignore[private-import]

    def _complete(
        self,
        data: dict,
    ) -> dict:
        prompt = data.pop("prompt")
        response = self.client.generate_content(prompt)  # type: ignore[private-import]
        completion = response.text if self.mode == "chat" else " ".join([r.text for r in response])

        prompt_tokens = len(prompt)
        completion_tokens = len(completion)
        cost_per_token = self.MODEL_INFO[self.model]
        cost = (
            (prompt_tokens * cost_per_token["prompt"]) + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000

        # fast approximation. We could call count_message_tokens() but this will add latency
        prompt_tokens = math.ceil((prompt_tokens + 1) / 4)
        completion_tokens = math.ceil((completion_tokens + 1) / 4)
        total_tokens = math.ceil(prompt_tokens + completion_tokens)

        return {
            "completion": completion,
            "model": self.model,
            "tokens": total_tokens,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
            "cost": cost,
        }
