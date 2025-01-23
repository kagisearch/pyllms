from __future__ import annotations

from dataclasses import dataclass

import ai21
from ai21.models.chat import ChatMessage
from ai21.tokenizers import get_tokenizer

from .base import SyncProvider, msg_as_str


@dataclass
class AI21Provider(SyncProvider):
    # per million tokens
    MODEL_INFO = {
        "j2-grande-instruct": {"prompt": 10.0, "completion": 10.0, "token_limit": 8192},
        "j2-jumbo-instruct": {"prompt": 15.0, "completion": 15.0, "token_limit": 8192},
    }

    def __post_init__(self):
        super().__post_init__()
        self.client = ai21.AI21Client(self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        return get_tokenizer(self.model + "-tokenizer").count_tokens(msg_as_str(content))

    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> dict:
        return {
            "prompt": prompt,
            "temperature": temperature,
            "maxTokens": max_tokens,
            **kwargs,
        }

    def _from_dict(self, data: dict) -> list[ChatMessage]:
        return [ChatMessage(role="user", content=data.pop("prompt"))]

    def _complete(self, data: dict) -> dict:
        response = self.client.chat.completions.create(model=self.model, messages=self._from_dict(data), **data)
        return {
            "completion": response.completions[0].data.text,
            "tokens_prompt": len(response.prompt.tokens),
            "tokens_completion": len(response.completions[0].data.tokens),
        }

    # TODO: async and stream support
