import time
from contextlib import contextmanager
from typing import Dict


class BaseProvider:
    """Base class for all providers.
    Methods will raise NotImplementedError if they are not overwritten.
    """

    MODEL_INFO = {}

    def __init__(self, model=None, api_key=None, **kwargs):
        self.latency = None
        # to be overwritten by subclasses
        self.api_key = api_key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.model}')"

    def __str__(self):
        return f"{self.__class__.__name__}('{self.model}')"

    def _prepare_model_inputs(
        self,
        **kwargs
    ) -> Dict:
        raise NotImplementedError()

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
            (prompt_tokens * cost_per_token["prompt"])
            + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000
        cost = round(cost, 5)
        return cost

    def count_tokens(self, content):
        raise NotImplementedError(
            f"Count tokens is currently not supported with {self.__class__.__name__}"
        )

    def complete(self, *args, **kwargs):
        raise NotImplementedError

    async def acomplete(self):
        raise NotImplementedError(
            f"Async complete is not yet supported with {self.__class__.__name__}"
        )

    def complete_stream(self):
        raise NotImplementedError(
            f"Streaming is not yet supported with {self.__class__.__name__}"
        )

    async def acomplete_stream(self):
        raise NotImplementedError(
            f"Async streaming is not yet supported with {self.__class__.__name__}"
        )
