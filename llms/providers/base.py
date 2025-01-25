from __future__ import annotations

import json
import time
import typing as t
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field


def msg_from_raw(cont: str | dict | list[dict], role: str = "user") -> list[dict]:
    if isinstance(cont, str):
        return [{"content": cont, "role": role}]
    if isinstance(cont, dict):
        return [cont]
    return cont


def msg_as_str(cont: list[dict]) -> str:
    return ";".join([f"{message['role'].capitalize()}: {message['content']}" for message in cont])


Provider = t.Union["AsyncProvider", "StreamProvider", "SyncProvider"]


@dataclass
class ABCResult(ABC):
    provider: Provider
    model_inputs: dict

    def __post_init__(self):
        self._meta = self._meta or {}
        self.function_call = self.function_call or {}
        self.text = self.text or ""

    @property
    def completion_tokens(self) -> int:
        if not (completion_tokens := self._meta.get("completion_tokens")):
            completion_tokens = self.provider.count_tokens(self.text)
            self._meta["completion_tokens"] = completion_tokens
        return completion_tokens

    @property
    def prompt_tokens(self) -> int:
        if not (prompt_tokens := self._meta.get("prompt_tokens")):
            prompt_tokens = self.provider.count_tokens(
                self.model_inputs.get("prompt") or self.model_inputs.get("messages") or ""
            )
            self._meta["prompt_tokens"] = prompt_tokens
        return prompt_tokens

    @property
    def tokens(self) -> int:
        return self.completion_tokens + self.prompt_tokens

    @property
    def cost(self) -> float:
        if not (cost := self._meta.get("cost")):
            cost = self.provider.compute_cost(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            )
            self._meta["cost"] = cost
        return cost

    @property
    def meta(self) -> dict:
        return {
            "model": self.provider.model,
            "tokens": self.tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
            "latency": self._meta.get("latency"),
            **self._meta,
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

    @abstractmethod
    def _count_tokens(self, content: list[dict]) -> int:
        pass

    def count_tokens(self, content: str | dict | list[dict]) -> int:
        return self._count_tokens(msg_from_raw(content))

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> dict:
        pass


@dataclass
class AsyncProvider(SyncProvider, ABC):
    @abstractmethod
    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        pass


@dataclass
class StreamProvider(AsyncProvider, ABC):
    @abstractmethod
    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        pass

    @abstractmethod
    def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        pass
