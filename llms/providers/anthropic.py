# llms/providers/anthropic.py

import itertools
import json
import os
import time
import aiohttp
import anthropic

from typing import Dict, List, Optional, Tuple, Union
from anthropic.api import _process_request_error
from .base_provider import BaseProvider


class AnthropicClient(anthropic.Client):
    """Extend Anthropic Client class to accept aiosession"""

    async def _arequest_as_json(
        self,
        method: str,
        path: str,
        params: dict,
        headers: Optional[Dict[str, str]] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        aiosession: Optional[aiohttp.ClientSession] = None,
    ) -> dict:
        if aiosession is None:
            super()._arequest_as_json(
                method=method,
                path=path,
                params=params,
                headers=headers,
                request_timeout=request_timeout,
            )
        else:
            request = self._request_params(
                headers, method, params, path, request_timeout
            )
            async with aiosession.request(
                request.method,
                request.url,
                headers=request.headers,
                data=request.data,
                timeout=request.timeout,
            ) as result:
                content = await result.text()
                if result.status != 200:
                    _process_request_error(method, content, result.status)
                json_body = json.loads(content)
                return json_body


class AnthropicProvider(BaseProvider):
    MODEL_INFO = {
        "claude-instant-v1": {"prompt": 1.63, "completion": 5.51, "token_limit": 9000},
        "claude-v1": {"prompt": 11.02, "completion": 32.68, "token_limit": 9000},
    }

    def __init__(self, api_key=None, model=None):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = AnthropicClient(api_key)

    def count_tokens(self, content: str):
        return anthropic.count_tokens(content)

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
        stop_sequences: Optional[List[str]] = None,
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

        if stop_sequences is None:
            stop_sequences = [anthropic.HUMAN_PROMPT]

        response = self.client.completion_stream(
            prompt=formatted_prompt,
            stop_sequences=stop_sequences,
            temperature=temperature,
            model=self.model,
            **kwargs,
        )

        last_completion = ""
        for data in response:
            new_chunk = data["completion"][len(last_completion) :]
            last_completion = data["completion"]
            yield (new_chunk)
