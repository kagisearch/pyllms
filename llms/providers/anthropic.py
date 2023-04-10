# llms/providers/anthropic.py

import os
import anthropic
import time


class AnthropicProvider:
    MODEL_INFO = {
        "claude-instant-v1": {"prompt": 0.43, "completion": 1.45, "token_limit": 8000},
        "claude-v1": {"prompt": 2.9, "completion": 8.6, "token_limit": 8000},
    }
    def __init__(self, api_key=None, model=None):
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key)

        if model is None:
            model = "claude-instant-v1"
        self.model = model

    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def complete(self, prompt,temperature=0, max_tokens_to_sample=200, **kwargs):
        formatted_prompt = f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}"
        start_time = time.time()
        response = self.client.completion(
            prompt=formatted_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            temperature=temperature, 
            max_tokens_to_sample=max_tokens_to_sample,
            model=self.model,
            **kwargs,
        )
        latency = time.time() - start_time
        completion = response["completion"].strip()

        # Calculate tokens and cost
        prompt_tokens = anthropic.count_tokens(formatted_prompt)
        completion_tokens = anthropic.count_tokens(completion)
        total_tokens = prompt_tokens + completion_tokens
        cost_per_token = self.MODEL_INFO[self.model]
        cost = (
            (prompt_tokens * cost_per_token["prompt"])
            + (completion_tokens * cost_per_token["completion"])
        ) / 1000000

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
