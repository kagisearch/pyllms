import inspect
import json
from typing import AsyncGenerator, Dict, Generator, List, Optional
from warnings import warn

from llms.providers.base_provider import BaseProvider


class Result:
    def __init__(
        self,
        text: str,
        model_inputs: Dict,
        provider: BaseProvider,
        meta: Optional[Dict] = None,
        function_call: Optional[Dict] = None,
    ):
        self._meta = meta or {}
        self.text = text
        self.provider = provider
        self.model_inputs = model_inputs
        self.function_call = function_call or {}

    @property
    def tokens_completion(self) -> int:
        if tokens_completion := self._meta.get("tokens_completion"):
            return tokens_completion
        else:
            tokens_completion = self.provider.count_tokens(self.text)
            self._meta["tokens_completion"] = tokens_completion
            return tokens_completion

    @property
    def tokens_prompt(self) -> int:
        if tokens_prompt := self._meta.get("tokens_prompt"):
            return tokens_prompt
        else:
            prompt = self.model_inputs.get("prompt") or self.model_inputs.get("messages")
            tokens_prompt = self.provider.count_tokens(prompt)
            self._meta["tokens_prompt"] = tokens_prompt
            return tokens_prompt

    @property
    def tokens(self) -> int:
        return self.tokens_completion + self.tokens_prompt

    @property
    def cost(self) -> float:
        if cost := self._meta.get("cost"):
            return cost
        else:
            cost = self.provider.compute_cost(
                prompt_tokens=self.tokens_prompt, completion_tokens=self.tokens_completion
            )
            self._meta["cost"] = cost
            return cost

    @property
    def meta(self) -> Dict:
        return {
            "model": self.provider.model,
            "tokens": self.tokens,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "cost": self.cost,
            "latency": self._meta.get("latency"),
        }

    def to_json(self):
        model_inputs = self.model_inputs
        # remove https related params
        model_inputs.pop("headers", None)
        model_inputs.pop("request_timeout", None)
        model_inputs.pop("aiosession", None)
        return json.dumps(
            {
                "text": self.text,
                "meta": self.meta,
                "model_inputs": model_inputs,
                "provider": str(self.provider),
                "function_call": self.function_call
            }
        )


class Results:
    def __init__(self, results: List[Result]):
        self._results = results

    @property
    def text(self):
        return [result.text for result in self._results]

    @property
    def meta(self):
        return [result.meta for result in self._results]

    def to_json(self):
        return json.dumps([result.to_json() for result in self._results])


class StreamResult:
    def __init__(
        self,
        stream: Generator,
        model_inputs: Dict,
        provider: BaseProvider,
        meta: Optional[Dict] = None,
    ):
        self._stream = stream
        self._meta = meta or {}
        self.provider = provider
        self.model_inputs = model_inputs

        self._streamed_text = []

    def __iter__(self):
        warn(
            "Looping through result will be deprecated, please loop through result.stream instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield from self.stream

    @property
    def stream(self):
        if not inspect.getgeneratorstate(self._stream) == "GEN_CLOSED":
            for item in self._stream:
                self._streamed_text.append(item)
                yield item
        else:
            yield from iter(self._streamed_text)

    @property
    def text(self) -> str:
        _ = all(self.stream)
        return "".join(self._streamed_text)

    @property
    def tokens_completion(self) -> int:
        if tokens_completion := self._meta.get("tokens_completion"):
            return tokens_completion
        else:
            tokens_completion = self.provider.count_tokens(self.text)
            self._meta["tokens_completion"] = tokens_completion
            return tokens_completion

    @property
    def tokens_prompt(self) -> int:
        if tokens_prompt := self._meta.get("tokens_prompt"):
            return tokens_prompt
        else:
            prompt = self.model_inputs.get("prompt") or self.model_inputs.get("messages")
            tokens_prompt = self.provider.count_tokens(prompt)
            self._meta["tokens_prompt"] = tokens_prompt
            return tokens_prompt

    @property
    def tokens(self) -> int:
        return self.tokens_completion + self.tokens_prompt

    @property
    def cost(self) -> float:
        if cost := self._meta.get("cost"):
            return cost
        else:
            cost = self.provider.compute_cost(
                prompt_tokens=self.tokens_prompt, completion_tokens=self.tokens_completion
            )
            self._meta["cost"] = cost
            return cost

    @property
    def meta(self) -> Dict:
        return {
            "model": self.provider.model,
            "tokens": self.tokens,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "cost": self.cost,
        }

    def to_json(self):
        model_inputs = self.model_inputs
        # remove https related params
        model_inputs.pop("headers", None)
        model_inputs.pop("request_timeout", None)
        return json.dumps(
            {
                "text": self.text,
                "meta": self.meta,
                "model_inputs": model_inputs,
                "provider": str(self.provider),
            }
        )


class AsyncIteratorWrapper:
    def __init__(self, obj):
        self._it = iter(obj)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            value = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return value


class AsyncStreamResult:
    def __init__(
        self,
        stream: AsyncGenerator,
        model_inputs: Dict,
        provider: BaseProvider,
        meta: Optional[Dict] = None,
    ):
        self._stream = stream
        self._meta = meta or {}
        self.provider = provider
        self.model_inputs = model_inputs

        self._stream_exhausted = False
        self._streamed_text = []

    def __aiter__(self):
        warn(
            "Looping through result will be deprecated, please loop through result.stream instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self

    async def __anext__(self):
        return await self._stream.__anext__()

    @property
    async def stream(self):
        if not self._stream_exhausted:
            async for item in self._stream:
                self._streamed_text.append(item)
                yield item
            self._stream_exhausted = True
        else:
            async for item in AsyncIteratorWrapper(self._streamed_text):
                yield item

    @property
    def text(self):
        if not self._stream_exhausted:
            raise RuntimeError("Please finish streaming the result.")
        return "".join(self._streamed_text)

    @property
    def tokens_completion(self) -> int:
        if tokens_completion := self._meta.get("tokens_completion"):
            return tokens_completion
        else:
            tokens_completion = self.provider.count_tokens(self.text)
            self._meta["tokens_completion"] = tokens_completion
            return tokens_completion

    @property
    def tokens_prompt(self) -> int:
        if tokens_prompt := self._meta.get("tokens_prompt"):
            return tokens_prompt
        else:
            prompt = self.model_inputs.get("prompt") or self.model_inputs.get("messages")
            tokens_prompt = self.provider.count_tokens(prompt)
            self._meta["tokens_prompt"] = tokens_prompt
            return tokens_prompt

    @property
    def tokens(self) -> int:
        return self.tokens_completion + self.tokens_prompt

    @property
    def cost(self) -> float:
        if cost := self._meta.get("cost"):
            return cost
        else:
            cost = self.provider.compute_cost(
                prompt_tokens=self.tokens_prompt, completion_tokens=self.tokens_completion
            )
            self._meta["cost"] = cost
            return cost

    @property
    def meta(self) -> Dict:
        return {
            "model": self.provider.model,
            "tokens": self.tokens,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "cost": self.cost,
        }

    def to_json(self):
        model_inputs = self.model_inputs
        # remove https related params
        model_inputs.pop("headers", None)
        model_inputs.pop("request_timeout", None)
        model_inputs.pop("aiosession", None)
        return json.dumps(
            {
                "text": self.text,
                "meta": self.meta,
                "model_inputs": model_inputs,
                "provider": str(self.provider),
            }
        )
