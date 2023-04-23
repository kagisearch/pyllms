# llms/providers/cohere.py

import itertools
import os
import cohere
import time

from typing import List, Optional


class CohereProvider:
    MODEL_INFO = {
        "command-xlarge-beta": {"prompt": 25.0, "completion": 25, "token_limit": 8192},
        "command-xlarge-nightly": {"prompt": 25.0, "completion": 25, "token_limit": 8192},
    }

    def __init__(self, api_key=None, model=None):
        if api_key is None:
            api_key = os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(api_key)

        if model is None:
            model = list(MODEL_INFO.keys())[0]
        self.model = model

    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def count_tokens(self, content: str):
        tokens = self.client.tokenize(content)
        return len(tokens)

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

        start_time = time.time()
        response = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model,
            **kwargs,
        )
        latency = time.time() - start_time

        completion = response.generations[0].text.strip()

        # Calculate tokens and cost
        # prompt_tokens = len(self.client.tokenize(prompt)) # too slow for normal use
        # completion_tokens =len(self.client.tokenize(completion))
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

    def complete_stream(
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

        if "stream" not in kwargs:
            kwargs["stream"] = True  # Add stream param if not present

        response = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model,
            **kwargs,
        )

        for token in response:
            yield token.text
