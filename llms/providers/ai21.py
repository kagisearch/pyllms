from __future__ import annotations

from dataclasses import dataclass

import ai21
from ai21.models.chat import ChatMessage
from ai21.tokenizers import get_tokenizer

from .base import ModelInfo, SyncProvider, msg_as_str


@dataclass
class AI21Provider(SyncProvider):
    # per million tokens
    MODEL_INFO = {
        "j2-grande-instruct": ModelInfo(prompt_cost=10.0, completion_cost=10.0, context_limit=8192),
        "j2-jumbo-instruct": ModelInfo(prompt_cost=15.0, completion_cost=15.0, context_limit=8192),
    }

    def __post_init__(self):
        super().__post_init__()
        self.client = ai21.AI21Client(self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        return get_tokenizer(self.model + "-tokenizer").count_tokens(msg_as_str(content))

    @staticmethod
    def prepare_input(
        **kwargs,
    ) -> dict:
        if max_tokens := kwargs.pop("max_tokens", None):
            kwargs["maxTokens"] = max_tokens
        return kwargs

    def complete(self, messages: list[dict], **kwargs) -> dict:
        data = self.prepare_input(**kwargs)
        response = self.client.chat.completions.create(
            model=self.model, messages=[ChatMessage(**ms) for ms in messages], **data
        )
        return {
            "completion": response.completions[0].data.text,
            "prompt_tokens": len(response.prompt.tokens),
            "completion_tokens": len(response.completions[0].data.tokens),
        }

    # TODO: async and stream support
