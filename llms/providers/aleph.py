# llms/providers/aleph.py

import os

import tiktoken
from aleph_alpha_client import AsyncClient, Client, CompletionRequest, Prompt

from ..results.result import Result
from .base_provider import BaseProvider


class AlephAlphaProvider(BaseProvider):
    MODEL_INFO = {
        "luminous-base": {"prompt": 6.6, "completion": 7.6, "token_limit": 2048},
        "luminous-extended": {"prompt": 9.9, "completion": 10.9, "token_limit": 2048},
        "luminous-supreme": {"prompt": 38.5, "completion": 42.5, "token_limit": 2048},
        "luminous-supreme-control": {
            "prompt": 48.5,
            "completion": 53.6,
            "token_limit": 2048,
        },
    }

    def __init__(self, api_key=None, model=None):
        if api_key is None:
            api_key = os.getenv("ALEPHALPHA_API_KEY")
        self.client = Client(api_key)
        self.async_client = AsyncClient(api_key)

        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model

    def count_tokens(self, content: str):
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(content))

    def _prepare_model_inputs(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> CompletionRequest:
        prompt = Prompt.from_text(prompt)
        maximum_tokens = kwargs.pop("maximum_tokens", max_tokens)

        model_inputs = CompletionRequest(
            prompt=prompt,
            temperature=temperature,
            maximum_tokens=maximum_tokens,
            **kwargs,
        )
        return model_inputs

    def complete(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> Result:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        with self.track_latency():
            response = self.client.complete(request=model_inputs, model=self.model)

        completion = response.completions[0].completion.strip()

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta={"latency": self.latency},
        )

    async def acomplete(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> Result:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        with self.track_latency():
            async with self.async_client as client:
                response = await client.complete(request=model_inputs, model=self.model)

        completion = response.completions[0].completion.strip()

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta={"latency": self.latency},
        )
