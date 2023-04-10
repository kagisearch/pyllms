# llms/providers/anthropic.py

import ai21
import time


class AI21Provider:
    # per million tokens
    MODEL_INFO = {
        "j2-jumbo-instruct": {"prompt": 15.0, "completion": 15.0, "token_limit": 8192},
        "j2-grande-instruct": {"prompt": 10.0, "completion": 10.0, "token_limit": 8192},
    }

    def __init__(self, api_key, model=None):
        ai21.api_key = api_key
        if model is None:
            model = "j2-grande-instruct"
        self.model = model

    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def complete(self, prompt, temperature=0, maxTokens=200, **kwargs):
        start_time = time.time()
        response = ai21.Completion.execute(model=self.model, prompt=prompt, temperature=temperature, maxTokens=maxTokens, **kwargs)
        latency = time.time() - start_time
        completion = response.completions[0].data.text.strip()
        prompt_tokens = len(response.prompt.tokens)
        completion_tokens = len(response.completions[0].data.tokens)
        total_tokens = prompt_tokens + completion_tokens

        cost_per_token = self.MODEL_INFO[self.model]
        cost = (prompt_tokens * cost_per_token["prompt"] / 1000000) + (
            completion_tokens * cost_per_token["completion"] / 1000000
        )

        return {
            "text": completion,
            "meta": {
                "model": self.model,
                "tokens": total_tokens,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
                "cost": cost,
                "latency": latency,
            },
        }
