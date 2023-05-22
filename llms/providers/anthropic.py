# llms/providers/anthropic.py

import json
import os
import aiohttp
import anthropic

from typing import AsyncIterator, Dict, List, Optional, Tuple, Union
from anthropic.api import _process_request_error
from .base_provider import BaseProvider


class AnthropicClient(anthropic.Client):
    """Extend Anthropic Client class to accept aiosession"""

    async def _arequest_as_json(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        aiosession: Optional[aiohttp.ClientSession] = None,
        **params: dict,
    ) -> dict:
        if aiosession is None:
            return await super()._arequest_as_json(
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

    async def _arequest_as_stream(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        aiosession: Optional[aiohttp.ClientSession] = None,
        **params: dict,
    ) -> AsyncIterator[dict]:
        # It seems there isn't async version of yield from
        # https://peps.python.org/pep-0525/#asynchronous-yield-from
        if aiosession is None:
            stream_outputs = super()._arequest_as_stream(
                method=method,
                path=path,
                params=params,
                headers=headers,
                request_timeout=request_timeout
            )
            async for output in stream_outputs:
                yield output
        else:
            request = self._request_params(headers, method, params, path, request_timeout)
            awaiting_ping_data = False
            async with aiosession.request(
                request.method,
                request.url,
                headers=request.headers,
                data=request.data,
                timeout=request.timeout,
            ) as result:
                if result.status != 200:
                    super()._process_request_error(method, await result.text(), result.status)
                async for line in result.content:
                    line = line.strip()
                    if not line:
                        continue
                    if line == b"event: ping":
                        awaiting_ping_data = True
                        continue
                    if awaiting_ping_data:
                        awaiting_ping_data = False
                        continue

                    if line == b"data: [DONE]":
                        continue

                    line = line.decode("utf-8")

                    prefix = "data: "
                    if line.startswith(prefix):
                        line = line[len(prefix) :]
                    yield json.loads(line)

    async def acompletion(self, **kwargs):
        # Override original method which pass kwargs as params.
        # We will pass kwargs in
        # _arequest_as_json will strips-off the keyword arguments that it needs
        # and pass the rest as params
        return await self._arequest_as_json("post", "/v1/complete", **kwargs)

    async def acompletion_stream(self, **kwargs) -> AsyncIterator:
        outputs = self._arequest_as_stream(
            "post",
            "v1/complete",
            **kwargs,
        )
        async for output in outputs:
            yield output


class AnthropicProvider(BaseProvider):
    MODEL_INFO = {
        "claude-instant-v1.1": {
            "prompt": 1.63,
            "completion": 5.51,
            "token_limit": 9000,
        },
        "claude-instant-v1": {"prompt": 1.63, "completion": 5.51, "token_limit": 9000},
        "claude-v1": {"prompt": 11.02, "completion": 32.68, "token_limit": 9000},
        "claude-v1-100k": {"prompt": 11.02, "completion": 32.68, "token_limit": 100000},
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

    def _prepare_model_input(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        stream: bool = False,
        **kwargs,
    ):
        formatted_prompt = (
            f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}{ai_prompt}"
        )

        if history is not None:
            history_text_list = []
            for message in history:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    role_prompt = anthropic.HUMAN_PROMPT
                elif role == "assistant":
                    role_prompt = anthropic.AI_PROMPT
                else:
                    raise ValueError(
                        f"Invalid role {role}, role must be user or assistant."
                    )

                formatted_message = f"{role_prompt}{content}"
                history_text_list.append(formatted_message)

            history_prompt = "".join(history_text_list)
            formatted_prompt = f"{history_prompt}{formatted_prompt}"

        max_tokens_to_sample = kwargs.pop("max_tokens_to_sample", max_tokens)

        if stop_sequences is None:
            stop_sequences = [anthropic.HUMAN_PROMPT]
        model_input = {
            "prompt": formatted_prompt,
            "temperature": temperature,
            "max_tokens_to_sample": max_tokens_to_sample,
            "stop_sequences": stop_sequences,
            "stream": stream,
            **kwargs,
        }
        return model_input

    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        **kwargs,
    ):
        """
        Args:
            history: messages in OpenAI format,
              each dict must include role and content key.
            ai_prompt: prefix of AI response, for finer control on the output.
        """

        model_input = self._prepare_model_input(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            **kwargs,
        )

        with self.track_latency():
            response = self.client.completion(model=self.model, **model_input)

        completion = response["completion"].strip()

        # Calculate tokens and cost
        prompt_tokens = anthropic.count_tokens(model_input["prompt"])
        completion_tokens = anthropic.count_tokens(response["completion"])
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

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        **kwargs,
    ):
        """
        Args:
            history: messages in OpenAI format,
              each dict must include role and content key.
            ai_prompt: prefix of AI response, for finer control on the output.
        """
        model_input = self._prepare_model_input(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            **kwargs,
        )
        with self.track_latency():
            response = await self.client.acompletion(model=self.model, **model_input)
        completion = response["completion"].strip()

        # Calculate tokens and cost
        prompt_tokens = anthropic.count_tokens(model_input["prompt"])
        completion_tokens = anthropic.count_tokens(response["completion"])
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

    def complete_stream(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        **kwargs,
    ):
        """
        Args:
            history: messages in OpenAI format,
              each dict must include role and content key.
            ai_prompt: prefix of AI response, for finer control on the output.
        """
        model_input = self._prepare_model_input(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            stream=True,
            **kwargs,
        )

        response = self.client.completion_stream(model=self.model, **model_input)

        first_completion = next(response)["completion"]
        yield first_completion.lstrip()

        last_completion = first_completion
        for data in response:
            new_chunk = data["completion"][len(last_completion) :]
            last_completion = data["completion"]
            yield (new_chunk)

    async def acomplete_stream(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        **kwargs,
    ) -> AsyncIterator:
        """
        Args:
            history: messages in OpenAI format,
              each dict must include role and content key.
            ai_prompt: prefix of AI response, for finer control on the output.
        """
        model_input = self._prepare_model_input(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            stream=True,
            **kwargs,
        )

        response = self.client.acompletion_stream(model=self.model, **model_input)
        first_completion = (await anext(response))["completion"]
        yield first_completion.lstrip()

        last_completion = first_completion
        async for data in response:
            new_chunk = data["completion"][len(last_completion) :]
            last_completion = data["completion"]
            yield (new_chunk)
