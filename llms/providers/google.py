# we could switch to genai  https://developers.generativeai.google/api/python/google/generativeai


import vertexai
from vertexai.preview.language_models import ChatModel

from typing import List, Optional


from .base_provider import BaseProvider

class GoogleProvider(BaseProvider):
    # cost is per million tokens
    MODEL_INFO = {
        "chat-bison": {
            "prompt": 0.5,
            "completion": 0.5,
            "token_limit": 0,
            "uses_characters": True,
        },
    }

    def __init__(self, model=None):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]

        self.model = model

        self.client = ChatModel.from_pretrained(model)

        vertexai.init()

    def _prepare_model_input(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ):
        temperature = max(temperature, 0.01)
        max_output_tokens = kwargs.pop("max_output_tokens", max_tokens)
        params = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            **kwargs,
        }
        return prompt, params

    def complete(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        context: str = None,
        examples: dict = {},
        **kwargs,
    ):
        prompt, params = self._prepare_model_input(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        with self.track_latency():
            chat = self.client.start_chat(context=context, examples=examples)
            response = chat.send_message(prompt, **params)

        completion = response.text

        # Calculate tokens and cost
        prompt_tokens = len(prompt)
        completion_tokens = len(completion)

        cost_per_token = self.MODEL_INFO[self.model]
        cost = (
            (prompt_tokens * cost_per_token["prompt"])
            + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000

        prompt_tokens = prompt_tokens / 4
        completion_tokens = completion_tokens / 4
        total_tokens = prompt_tokens + completion_tokens

        return {
            "text": completion,
            "meta": {
                "model": self.model,
                "tokens": total_tokens,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
                "cost": cost,
                "latency": self.latency,
            },
            "provider": str(self),
        }
