from __future__ import annotations

import itertools as it
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

    def _count_tokens(self, content: list[dict]) -> int:
        return self.client.count_tokens(msg_as_str(content)).total_tokens  # type: ignore[private-import]

    @staticmethod
    def prepare_input(
        **kwargs,
    ) -> dict:
        if max_tokens := kwargs.pop("max_tokens"):
            kwargs["max_output_tokens"] = max_tokens
        return kwargs

    def complete(self, messages: list[dict], **kwargs) -> dict:
        prompts = [
            {"role": parts[0]["role"], "parts": [p["content"] for p in parts]}
            for parts in (list(ps) for _, ps in it.groupby(messages, key=lambda x: x["role"]))
        ]
        kwargs = self.prepare_input(**kwargs)
        response = self.client.generate_content(prompts, **kwargs)  # type: ignore[private-import]
        completion = response.text if self.mode == "chat" else " ".join([r.text for r in response])

        prompt_tokens = len(msg_as_str(messages))
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }
