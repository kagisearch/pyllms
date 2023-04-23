import itertools
import openai
import tiktoken
import time
from typing import List, Optional


class OpenAIProvider:
    # cost is per million tokens
    MODEL_INFO = {
        "gpt-3.5-turbo": {"prompt": 2.0, "completion": 2.0, "token_limit": 4000},
        "gpt-4": {"prompt": 30.0, "completion": 60.0, "token_limit": 8000},
    }

    def __init__(self, api_key, model=None):
        openai.api_key = api_key
        if model is None:
            model = list(MODEL_INFO.keys())[0]
        self.model = model

    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def count_tokens(self, content: str):
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(content))

    def complete(
        self,
        prompt: str,
        history: Optional[List[tuple]] = None,
        system_message: str = None,
        temperature: float = 0,
        **kwargs,
    ):
        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]

        if history:
            role_cycle = itertools.cycle(("user", "assistant"))
            history_messages = itertools.chain.from_iterable(history)

            history = [
                {"role": role, "content": message}
                for role, message in zip(role_cycle, history_messages)
                if message is not None
            ]
            messages = [*history, *messages]

        if system_message:
            messages = [{"role": "system", "content": system_message}, *messages]

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

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[tuple]] = None,
        system_message: str = None,
        temperature: float = 0,
        **kwargs,
    ):
        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]

        if history:
            role_cycle = itertools.cycle(("user", "assistant"))
            history_messages = itertools.chain.from_iterable(history)

            history = [
                {"role": role, "content": message}
                for role, message in zip(role_cycle, history_messages)
                if message is not None
            ]
            messages = [*history, *messages]

        if system_message:
            messages = [{"role": "system", "content": system_message}, *messages]

        response = await openai.ChatCompletion.acreate(
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

    def complete_stream(
        self,
        prompt: str,
        history: Optional[List[tuple]] = None,
        system_message: str = None,
        temperature: float = 0,
        **kwargs,
    ):
        messages = [{"role": "user", "content": prompt}]

        if history:
            role_cycle = itertools.cycle(("user", "assistant"))
            history_messages = itertools.chain.from_iterable(history)

            history = [
                {"role": role, "content": message}
                for role, message in zip(role_cycle, history_messages)
                if message is not None
            ]
            messages = [*history, *messages]

        if system_message:
            messages = [{"role": "system", "content": system_message}, *messages]

        if "stream" not in kwargs:
            kwargs["stream"] = True  # Add stream param if not present

        response = openai.ChatCompletion.create(
            model=self.model, messages=messages, temperature=temperature, **kwargs
        )

        yield from (
            chunk["choices"][0].get("delta", {}).get("content") for chunk in response
        )
