# llms/providers/anthropic.py

import itertools
import os
import anthropic
import time

from typing import List, Optional


class AnthropicProvider:
    MODEL_INFO = {
        "claude-instant-v1": {"prompt": 1.63, "completion": 5.51, "token_limit": 9000},
        "claude-v1": {"prompt": 11.02, "completion": 32.68, "token_limit": 9000},
    }

    def __init__(self, api_key=None, model=None):
        
        if model is None:
            model = list(MODEL_INFO.keys())[0]
        self.model = model

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key)


    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def count_tokens(self, content: str):
        raise ValueError("Count tokens is currently not supported with AI21")

    def complete(
        self,
        prompt: str,
        history: Optional[List[tuple]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        **kwargs,
    ):
        formatted_prompt = (
            f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}{ai_prompt}"
        )
        if history is not None:
            role_cycle = itertools.cycle((anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT))
            history_messages = itertools.chain.from_iterable(history)
            history_prompt = "".join(
                itertools.chain.from_iterable(zip(role_cycle, history_messages))
            )
            formatted_prompt = f"{history_prompt}{formatted_prompt}"

        if "max_tokens_to_sample" not in kwargs:
            kwargs[
                "max_tokens_to_sample"
            ] = max_tokens  # Add maxTokens to kwargs if not present

        if stop_sequences is None:
            stop_sequences = [anthropic.HUMAN_PROMPT]

        start_time = time.time()
        response = self.client.completion(
            prompt=formatted_prompt,
            temperature=temperature,
            model=self.model,
            stop_sequences=stop_sequences,
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

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[tuple]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        **kwargs,
    ):
        formatted_prompt = (
            f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}{ai_prompt}"
        )
        if history is not None:
            role_cycle = itertools.cycle((anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT))
            history_messages = itertools.chain.from_iterable(history)
            history_prompt = "".join(
                itertools.chain.from_iterable(zip(role_cycle, history_messages))
            )
            formatted_prompt = f"{history_prompt}{formatted_prompt}"

        if "max_tokens_to_sample" not in kwargs:
            kwargs[
                "max_tokens_to_sample"
            ] = max_tokens  # Add maxTokens to kwargs if not present

        if stop_sequences is None:
            stop_sequences = [anthropic.HUMAN_PROMPT]

        start_time = time.time()
        response = await self.client.acompletion(
            prompt=formatted_prompt,
            temperature=temperature,
            model=self.model,
            stop_sequences=stop_sequences,
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
        formatted_prompt = f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}"
        if history is not None:
            role_cycle = itertools.cycle((anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT))
            history_messages = itertools.chain.from_iterable(history)
            history_prompt = "".join(
                itertools.chain.from_iterable(zip(role_cycle, history_messages))
            )
            formatted_prompt = f"{history_prompt}{formatted_prompt}"

        if "max_tokens_to_sample" not in kwargs:
            kwargs[
                "max_tokens_to_sample"
            ] = max_tokens  # Add maxTokens to kwargs if not present

        if "stream" not in kwargs:
            kwargs["stream"] = True  # Add stream param if not present

        response = self.client.completion_stream(
            prompt=formatted_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            temperature=temperature,
            model=self.model,
            **kwargs,
        )

        last_completion = ""
        for data in response:
            new_chunk = data["completion"][len(last_completion) :]
            last_completion = data["completion"]
            yield (new_chunk)
