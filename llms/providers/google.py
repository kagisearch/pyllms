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

from .base import ModelInfo, SyncProvider


@dataclass
class GoogleProvider(SyncProvider):
    # cost is per million tokens
    MODEL_INFO = {
        # no support for "textembedding-gecko"
        "chat-bison": ModelInfo(prompt_cost=0.5, completion_cost=0.5, context_limit=0),
        "text-bison": ModelInfo(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
        "text-bison-32k": ModelInfo(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
        "code-bison": ModelInfo(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
        "code-bison-32k": ModelInfo(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
        "codechat-bison": ModelInfo(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
        "codechat-bison-32k": ModelInfo(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
        "gemini-pro": ModelInfo(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
        "gemini-1.5-pro-preview-0514": ModelInfo(
            prompt_cost=0.35,
            completion_cost=0.53,
            context_limit=0,
            quirks={
                "uses_characters": False,
            },
        ),
        "gemini-1.5-flash-preview-0514": ModelInfo(
            prompt_cost=0.35,
            completion_cost=0.53,
            context_limit=0,
            quirks={
                "uses_characters": False,
            },
        ),
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

    def _count_tokens(self, content: list[dict]) -> int:
        raise

    @staticmethod
    def prepare_input(
        **kwargs,
    ) -> dict:
        if max_tokens := kwargs.pop("max_tokens"):
            kwargs["max_output_tokens"] = max_tokens
        return kwargs

    def complete(self, messages: list[dict], **kwargs) -> dict:
        kwargs = self.prepare_input(**kwargs)
        prompt = kwargs.pop(self.prompt_key, None) or messages[0]["content"]
        if isinstance(self.client, GenerativeModel):
            chat = self.client.start_chat()
            response = chat.send_message([prompt], generation_config=kwargs)
        elif isinstance(self.client, (ChatModel, CodeChatModel)):
            chat = self.client.start_chat()
            response = chat.send_message(**kwargs)
        else:  # text / code
            response = self.client.predict(**kwargs)

        completion = response.text or ""

        # Calculate tokens and cost
        if self.info.quirks.get("uses_characters", True):
            prompt_tokens = len(prompt)
            completion_tokens = len(completion)
        else:
            prompt_tokens = len(prompt) / 4
            completion_tokens = len(completion) / 4

        cost = ((prompt_tokens * self.info.prompt_cost) + (completion_tokens * self.info.completion_cost)) / 1_000_000

        if not self.info.quirks.get("uses_characters", True):
            prompt_tokens = math.ceil((prompt_tokens + 1) / 4)
            completion_tokens = math.ceil((completion_tokens + 1) / 4)
        total_tokens = prompt_tokens + completion_tokens

        return {
            "completion": completion,
            "model": self.model,
            "tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }
