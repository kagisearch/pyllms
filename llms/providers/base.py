from __future__ import annotations

import json
import time
import typing as t
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field


def from_raw(cont: str | dict | list[dict], role: str = "user") -> list[dict]:
    if isinstance(cont, str):
        return [{"content": cont, "role": role}]
    if isinstance(cont, dict):
        return [cont]
    return cont


def msg_as_str(cont: list[dict]) -> str:
    return ";".join([f"{message['role']}{message['content']}" for message in cont])


Provider = t.Union["AsyncProvider", "StreamProvider", "SyncProvider"]


@dataclass
class ABCResult(ABC):
    provider: Provider
    model_inputs: dict
    _meta: dict
    function_call: dict

    def __post_init__(self):
        self._meta = self._meta or {}
        self.function_call = self.function_call or {}
        self.text = self.text or ""

    @property
    def tokens_completion(self) -> int:
        if tokens_completion := self._meta.get("tokens_completion"):
            return tokens_completion
        tokens_completion = self.provider.count_tokens(self.text)
        self._meta["tokens_completion"] = tokens_completion
        return tokens_completion

    @property
    def tokens_prompt(self) -> int:
        if tokens_prompt := self._meta.get("tokens_prompt"):
            return tokens_prompt
        prompt: str = self.model_inputs.get("prompt") or self.model_inputs.get("messages") or ""
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
        cost = self.provider.compute_cost(
            prompt_tokens=self.tokens_prompt,
            completion_tokens=self.tokens_completion,
        )
        self._meta["cost"] = cost
        return cost

    @property
    def meta(self) -> dict:
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
                "function_call": self.function_call,
            }
        )


@dataclass
class Result(ABCResult):
    text: str
    _meta: dict = field(default_factory=dict)
    function_call: dict = field(default_factory=dict)


@dataclass
class StreamResult(ABCResult):
    _stream: t.Iterator
    _streamed_text: list = field(default_factory=list)
    _meta: dict = field(default_factory=dict)
    function_call: dict = field(default_factory=dict)

    @property
    def stream(self):
        while t := next(self._stream, None):
            self._streamed_text.append(t)
            yield t

    @property
    def text(self) -> str:
        _ = all(self.stream)
        return "".join(self._streamed_text)


@dataclass
class AsyncStreamResult(ABCResult):
    _stream: t.AsyncIterable
    _stream_exhausted: bool = False
    _streamed_text: list = field(default_factory=list)
    _meta: dict = field(default_factory=dict)
    function_call: dict = field(default_factory=dict)

    @property
    async def stream(self) -> t.AsyncIterator[Result]:
        if not self._stream_exhausted:
            async for item in self._stream:
                self._streamed_text.append(item)
                yield item
            self._stream_exhausted = True
        else:
            for r in self._streamed_text:
                yield r

    @property
    def text(self):
        if not self._stream_exhausted:
            msg = "Please finish streaming the result."
            raise RuntimeError(msg)
        return "".join(self._streamed_text)


@dataclass
class SyncProvider(ABC):
    """Base class for all providers.
    Methods will raise NotImplementedError if they are not overwritten.
    """

    api_key: str
    model: t.Any = None
    latency: float | None = None
    MODEL_INFO: t.ClassVar[dict] = {}

    def __post_init__(self):
        self.model = self.model or list(self.MODEL_INFO.keys())[0]

    @abstractmethod
    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 300,
        *args,
        **kwargs,
    ) -> dict:
        pass

    @abstractmethod
    def _complete(self, data: dict) -> dict:
        pass

    @abstractmethod
    def _count_tokens(self, content: list[dict]) -> int:
        pass

    def count_tokens(self, content: str | dict | list[dict]) -> int:
        return self._count_tokens(from_raw(content))

    @contextmanager
    def track_latency(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.latency = round(time.perf_counter() - start, 2)

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost_per_token = self.MODEL_INFO[self.model]
        cost = (
            (prompt_tokens * cost_per_token["prompt"]) + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000
        return round(cost, 5)

    def complete(
        self,
        prompt: str,
        *args,
        **kwargs,
    ) -> Result:
        model_inputs = self._prepare_input(prompt, *args, **kwargs)
        with self.track_latency():
            response = self._complete(model_inputs)

        completion = response.pop("completion")
        function_call = response.pop("function_call", None)

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            _meta={"latency": self.latency, **response},
            function_call=function_call,
        )


@dataclass
class AsyncProvider(SyncProvider, ABC):
    @abstractmethod
    async def _acomplete(self, data: dict) -> dict:
        pass

    async def acomplete(
        self,
        *args,
        **kwargs,
    ) -> Result:
        model_inputs = self._prepare_input(*args, **kwargs)

        with self.track_latency():
            response = await self._acomplete(model_inputs)

        completion = response.pop("completion")
        function_call = response.pop("function_call", None)

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            _meta={"latency": self.latency, **response},
            function_call=function_call,
        )


@dataclass
class StreamProvider(AsyncProvider, ABC):
    @abstractmethod
    def _complete_stream(self, data: dict) -> t.Iterator[str]:
        pass

    def complete_stream(self, *args, **kwargs) -> StreamResult:
        model_inputs = self._prepare_input(*args, **kwargs)

        return StreamResult(
            _stream=self._complete_stream(model_inputs),
            model_inputs=model_inputs,
            provider=self,
        )

    @abstractmethod
    def _acomplete_stream(self, data: dict) -> t.AsyncIterator[str]:
        pass

    def acomplete_stream(self, *args, **kwargs) -> AsyncStreamResult:
        model_inputs = self._prepare_input(*args, **kwargs)

        return AsyncStreamResult(
            _stream=self._acomplete_stream(model_inputs),
            model_inputs=model_inputs,
            provider=self,
        )
