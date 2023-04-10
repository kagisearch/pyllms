import openai
import time


class OpenAIProvider:
    # per million tokens
    MODEL_INFO = {
        "gpt-4": {"prompt": 30.0, "completion": 60.0, "token_limit": 8000},
        "gpt-3.5-turbo": {"prompt": 2.0, "completion": 2.0, "token_limit": 4000},
    }
    def __init__(self, api_key, model=None):
        openai.api_key = api_key
        if model is None:
            model = "gpt-3.5-turbo"
        self.model = model

    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def complete(self, prompt, temperature=0, system=None, **kwargs):
        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]

        if system:
            messages=[{"role": "system", "content": system}]+messages

        response = openai.ChatCompletion.create(
            model=self.model, messages=messages, temperature=temperature, **kwargs
        )

        latency = time.time() - start_time
        completion = response.choices[0].message.content.strip()
        usage = response.usage
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]

        cost_per_token = self.MODEL_INFO[self.model]
        cost = (prompt_tokens * cost_per_token["prompt"] / 1000000) + (
            completion_tokens * cost_per_token["completion"] / 1000000
        )

        return {
            "text": completion,
            "meta": {
                "model": self.model,
                "tokens": total_tokens,
                "tokens_prompt": prompt_tokens,  # Add tokens_prompt to meta
                "tokens_completion": completion_tokens,  # Add tokens_completion to meta
                "cost": cost,
                "latency": latency,
            },
        }
