# llms/providers/cohere.py

import os
from typing import Dict, Generator

import cohere

from ..results.result import Result, StreamResult
from .base_provider import BaseProvider


class CohereProvider(BaseProvider):
    MODEL_INFO = {
        "command": {"prompt": 15.0, "completion": 15, "token_limit": 2048},
        "command-nightly": {
            "prompt": 15.0,
            "completion": 15,
            "token_limit": 4096,
        },
    }

    def __init__(self, api_key=None, model=None):
        if api_key is None:
            api_key = os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(api_key)
        self.async_client = cohere.AsyncClient(api_key)

        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model

    def count_tokens(self, content: str) -> int:
        tokens = self.client.tokenize(content)
        return len(tokens)

    def _prepare_model_inputs(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        stream: bool = False,
        **kwargs,
    ) -> Dict:
        model_inputs = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        return model_inputs

    def complete(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> Result:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        with self.track_latency():
            response = self.client.generate(
                model=self.model,
                **model_inputs,
            )

        completion = response.generations[0].text.strip()
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
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        with self.track_latency():
            async with self.async_client() as client:
                response = await client.generate(
                    model=self.model,
                    **model_inputs,
                )

        completion = response.generations[0].text.strip()

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta={"latency": self.latency},
        )

    def complete_stream(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        response = self.client.generate(
            model=self.model,
            **model_inputs,
        )

        stream = self._process_stream(response)
        return StreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    def _process_stream(self, response: Generator) -> Generator:
        first_text = next(response).text
        yield first_text.lstrip()

        for token in response:
            yield token.text
