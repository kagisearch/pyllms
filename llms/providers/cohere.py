# llms/providers/cohere.py

import os

import cohere

from .base_provider import BaseProvider


class CohereProvider(BaseProvider):
    MODEL_INFO = {
        "command": {"prompt": 25.0, "completion": 25, "token_limit": 2048},
        "command-nightly": {
            "prompt": 25.0,
            "completion": 25,
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

    def count_tokens(self, content: str):
        tokens = self.client.tokenize(content)
        return len(tokens)

    def _prepare_model_input(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        stream: bool = False,
        **kwargs,
    ):
        model_input = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        return model_input

    def complete(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        model_input = self._prepare_model_input(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        with self.track_latency():
            response = self.client.generate(
                model=self.model,
                **model_input,
            )

        completion = response.generations[0].text.strip()

        # Calculate tokens and cost
        prompt_tokens = self.count_tokens(prompt) # too slow for normal use
        completion_tokens = self.count_tokens(completion)
        #prompt_tokens = -1
        #completion_tokens = -1
        total_tokens = prompt_tokens + completion_tokens
        cost = self.compute_cost(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

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

    async def acomplete(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        model_input = self._prepare_model_input(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        with self.track_latency():
            async with self.async_client() as client:
                response = await client.generate(
                    model=self.model,
                    **model_input,
                )

        completion = response.generations[0].text.strip()

        # Calculate tokens and cost
        # prompt_tokens = len(self.client.tokenize(prompt)) # too slow for normal use
        # completion_tokens =len(self.client.tokenize(completion))
        prompt_tokens = -1
        completion_tokens = -1
        total_tokens = prompt_tokens + completion_tokens
        cost = self.compute_cost(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

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

    def complete_stream(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        model_input = self._prepare_model_input(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        response = self.client.generate(
            model=self.model,
            **model_input,
        )

        first_text = next(response)
        yield first_text.text.lstrip()

        for token in response:
            yield token.text
