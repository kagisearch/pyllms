# llms/providers/cohere.py

import itertools
import os
from aleph_alpha_client import Client, CompletionRequest, Prompt
import time

from typing import List, Optional


class AlephAlphaProvider:
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

        if model is None:
            model = list(MODEL_INFO.keys())[0]
        self.model = model

    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def complete(
        self,
        prompt: str,
        history: Optional[List[tuple]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        if history is not None:
            HUMAN_PROMPT = "\n\nHuman:"
            AI_PROMPT = "\n\nAssistant:"
            role_cycle = itertools.cycle((HUMAN_PROMPT, AI_PROMPT))
            history_messages = itertools.chain.from_iterable(history)
            history_prompt = "".join(
                itertools.chain.from_iterable(zip(role_cycle, history_messages))
            )
            prompt = f"{history_prompt}{prompt}"

        if "maximum_tokens" not in kwargs:
            kwargs["maximum_tokens"] = max_tokens

        start_time = time.time()
        response = self.client.complete(
            CompletionRequest(prompt=Prompt.from_text(prompt), **kwargs),
            model=self.model,
        )

        latency = time.time() - start_time

        completion = response.completions[0].completion.strip()

        # Calculate tokens and cost
        prompt_tokens = -1
        completion_tokens = -1

        total_tokens = prompt_tokens + completion_tokens
        cost_per_token = self.MODEL_INFO[self.model]
        cost = (
            (prompt_tokens * cost_per_token["prompt"])
            + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000

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
