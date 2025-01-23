from __future__ import annotations

# NOTE: we could switch to genai  https://developers.generativeai.google/api/python/google/generativeai
import math
from dataclasses import dataclass

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import (
    ChatModel,
    CodeChatModel,
    CodeGenerationModel,
    TextGenerationModel,
)

from .base import SyncProvider


@dataclass
class GoogleProvider(SyncProvider):
    api_key = ""
    # cost is per million tokens
    MODEL_INFO = {
        # no support for "textembedding-gecko"
        "chat-bison": {
            "prompt": 0.5,
            "completion": 0.5,
            "token_limit": 0,
            "uses_characters": True,
        },
        "text-bison": {
            "prompt": 1.0,
            "completion": 1.0,
            "token_limit": 0,
            "uses_characters": True,
        },
        "text-bison-32k": {
            "prompt": 1.0,
            "completion": 1.0,
            "token_limit": 0,
            "uses_characters": True,
        },
        "code-bison": {
            "prompt": 1.0,
            "completion": 1.0,
            "token_limit": 0,
            "uses_characters": True,
        },
        "code-bison-32k": {
            "prompt": 1.0,
            "completion": 1.0,
            "token_limit": 0,
            "uses_characters": True,
        },
        "codechat-bison": {
            "prompt": 1.0,
            "completion": 1.0,
            "token_limit": 0,
            "uses_characters": True,
        },
        "codechat-bison-32k": {
            "prompt": 1.0,
            "completion": 1.0,
            "token_limit": 0,
            "uses_characters": True,
        },
        "gemini-pro": {
            "prompt": 1.0,
            "completion": 1.0,
            "token_limit": 0,
            "uses_characters": True,
        },
        "gemini-1.5-pro-preview-0514": {
            "prompt": 0.35,
            "completion": 0.53,
            "token_limit": 0,
            "uses_characters": False,
        },
        "gemini-1.5-flash-preview-0514": {
            "prompt": 0.35,
            "completion": 0.53,
            "token_limit": 0,
            "uses_characters": False,
        },
    }

    def __post_init__(self):
        super().__post_init__()
        model = self.model

        if model.startswith("text-"):
            self.client = TextGenerationModel.from_pretrained(model)
            self.prompt_key = "prompt"
        elif model.startswith("code-"):
            self.client = CodeGenerationModel.from_pretrained(model)
            self.prompt_key = "prefix"
        elif model.startswith("codechat-"):
            self.client = CodeChatModel.from_pretrained(model)
            self.prompt_key = "message"
        elif model.startswith("gemini"):
            self.client = GenerativeModel(model)
            self.prompt_key = "message"
        else:
            self.client = ChatModel.from_pretrained(model)
            self.prompt_key = "message"

        vertexai.init()

    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ) -> dict:
        temperature = max(temperature, 0.01)
        if isinstance(self.client, GenerativeModel):
            return {
                "prompt": prompt,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                **kwargs,
            }

        return {
            self.prompt_key: prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        prompt = data.pop("prompt") or data.pop(self.prompt_key)
        if isinstance(self.client, GenerativeModel):
            chat = self.client.start_chat()
            response = chat.send_message([prompt], generation_config=data)
        elif isinstance(self.client, (ChatModel, CodeChatModel)):
            chat = self.client.start_chat()
            response = chat.send_message(**data)
        else:  # text / code
            response = self.client.predict(**data)

        completion = response.text or ""

        cost_per_token = self.MODEL_INFO[self.model]

        # Calculate tokens and cost
        if cost_per_token["uses_characters"]:
            prompt_tokens = len(prompt)
            completion_tokens = len(completion)
        else:
            prompt_tokens = len(prompt) / 4
            completion_tokens = len(completion) / 4

        cost = (
            (prompt_tokens * cost_per_token["prompt"]) + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000

        if not cost_per_token["uses_characters"]:
            prompt_tokens = math.ceil((prompt_tokens + 1) / 4)
            completion_tokens = math.ceil((completion_tokens + 1) / 4)
        total_tokens = prompt_tokens + completion_tokens

        return {
            "completion": completion,
            "model": self.model,
            "tokens": total_tokens,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
            "cost": cost,
        }
