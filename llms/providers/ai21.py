# llms/providers/ai21.py

import ai21

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

    def _prepare_model_input(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        maxTokens = kwargs.pop("maxTokens", max_tokens)
        model_input = {
            "prompt": prompt,
            "temperature": temperature,
            "maxTokens": maxTokens,
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
            response = ai21.Completion.execute(model=self.model, **model_input)

        completion = response.completions[0].data.text.strip()
        prompt_tokens = len(response.prompt.tokens)
        completion_tokens = len(response.completions[0].data.tokens)
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
