from __future__ import annotations

import typing as t
import warnings
from dataclasses import dataclass, field

from ollama import AsyncClient, Client

from .base import ModelInfo, StreamProvider, msg_as_str


def _get_model_info(ollama_host: str | None = "http://localhost:11434"):
    model_info = {}
    try:
        pulled_models = Client(host=ollama_host).list().get("models", [])
        for model in pulled_models:
            name = model["model"]
            # Ollama models are free to use locally
            model_info[name] = ModelInfo(
                prompt_cost=0.0,
                completion_cost=0.0,
                context_limit=4096,  # Default token limit
            )

        if not pulled_models:
            msg = "Could not retrieve any models from Ollama"
            raise ValueError(msg)
    except Exception as e:
        warnings.warn(f"Could not connect to Ollama server: {str(e)}", stacklevel=2)

    return model_info


@dataclass
class OllamaProvider(StreamProvider):
    api_key = ""
    MODEL_INFO = _get_model_info()

    ollama_host: str | None = "http://localhost:11434"
    ollama_client_options: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.client = Client(host=self.ollama_host, **self.ollama_client_options)
        self.async_client = AsyncClient(host=self.ollama_host, **self.ollama_client_options)

    def _count_tokens(self, content: list[dict]) -> int:
        """Estimate token count using simple word-based heuristic"""
        # Rough estimation: split on whitespace
        # TODO: also split on punctuation
        return len(msg_as_str(content).split())

    def complete(self, messages: list[dict], **kwargs) -> dict:
        try:
            response = self.client.chat(model=self.model, messages=messages, stream=False, **kwargs)
        except Exception as e:
            msg = f"Ollama completion failed: {str(e)}"
            raise RuntimeError(msg) from e

        return {
            "completion": response.message.content,
            "prompt_tokens": response.prompt_eval_count,
            "completion_tokens": response.eval_count,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        try:
            response = await self.async_client.chat(model=self.model, messages=messages, stream=False, **kwargs)
        except Exception as e:
            msg = f"Ollama completion failed: {str(e)}"
            raise RuntimeError(msg) from e

        return {
            "completion": response.message.content,
            "prompt_tokens": response.prompt_eval_count,
            "completion_tokens": response.eval_count,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        for chunk in self.client.chat(model=self.model, messages=messages, stream=True, **kwargs):
            if c := chunk["message"]["content"]:
                yield c

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        async for chunk in await self.async_client.chat(model=self.model, messages=messages, stream=True, **kwargs):
            if c := chunk["message"]["content"]:
                yield c
