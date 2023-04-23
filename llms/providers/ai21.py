# llms/providers/anthropic.py

import ai21
import time
import itertools
from typing import List, Optional


class AI21Provider:
    # per million tokens
    MODEL_INFO = {
        "j2-grande-instruct": {"prompt": 10.0, "completion": 10.0, "token_limit": 8192},
        "j2-jumbo-instruct": {"prompt": 15.0, "completion": 15.0, "token_limit": 8192},
    }

    def __init__(self, api_key, model=None):
        ai21.api_key = api_key
        if model is None:
            model = list(MODEL_INFO.keys())[0]
        self.model = model

    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def count_tokens(self, content: str):
        return anthropic.count_tokens(content)

    def complete(
        self,
        prompt: str,
        history: Optional[List[tuple]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        HUMAN_PROMPT = "\n\nHuman:"
        AI_PROMPT = "\n\nAssistant:"

        if history is not None:
            role_cycle = itertools.cycle((HUMAN_PROMPT, AI_PROMPT))
            history_messages = itertools.chain.from_iterable(history)
            history_prompt = "".join(
                itertools.chain.from_iterable(zip(role_cycle, history_messages))
            )
            prompt = f"{history_prompt}{prompt}"

        if "maxTokens" not in kwargs:
            kwargs["maxTokens"] = max_tokens  # Add maxTokens to kwargs if not present

        start_time = time.time()
        response = ai21.Completion.execute(
            model=self.model, prompt=prompt, temperature=temperature, **kwargs
        )
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
