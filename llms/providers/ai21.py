# llms/providers/ai21.py

import ai21

from ..results.result import Result
from .base_provider import BaseProvider


class AI21Provider(BaseProvider):
    # per million tokens
    MODEL_INFO = {
        "j2-grande-instruct": {"prompt": 10.0, "completion": 10.0, "token_limit": 8192},
        "j2-jumbo-instruct": {"prompt": 15.0, "completion": 15.0, "token_limit": 8192},
    }

    def __init__(self, api_key, model=None):
        ai21.api_key = api_key
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model

    def _prepare_model_inputs(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        maxTokens = kwargs.pop("maxTokens", max_tokens)
        model_inputs = {
            "prompt": prompt,
            "temperature": temperature,
            "maxTokens": maxTokens,
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
            response = ai21.Completion.execute(model=self.model, **model_inputs)

        completion = response.completions[0].data.text.strip()
        tokens_prompt = len(response.prompt.tokens)
        tokens_completion = len(response.completions[0].data.tokens)

        meta = {
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "latency": self.latency,
        }

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta=meta,
        )
