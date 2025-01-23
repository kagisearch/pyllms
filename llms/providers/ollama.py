from __future__ import annotations

import typing as t
import warnings
from dataclasses import dataclass, field

from ollama import AsyncClient, Client

from .base import StreamProvider, msg_as_str


def _get_model_info(ollama_host: str | None = "http://localhost:11434"):
    model_info = {}
    try:
        pulled_models = Client(host=ollama_host).list().get("models", [])
        for model in pulled_models:
            name = model["model"]
            # Ollama models are free to use locally
            model_info[name] = {
                "prompt": 0.0,
                "completion": 0.0,
                "token_limit": 4096,  # Default token limit
            }

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

    def _count_tokens(self, content: list[dict]) -> int:
        """Estimate token count using simple word-based heuristic"""
        # Rough estimation: split on whitespace
        # TODO: also split on punctuation
        return len(msg_as_str(content).split())

    def __post_init__(self):
        super().__post_init__()
        self.client = Client(host=self.ollama_host, **self.ollama_client_options)
        self.async_client = AsyncClient(host=self.ollama_host, **self.ollama_client_options)

    def _prepare_input(
        self,
        prompt: str,
        history: list[dict] | None = None,
        system_message: str | list[dict] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        # Remove unsupported parameters
        kwargs.pop("max_tokens", None)
        kwargs.pop("temperature", None)
        messages = [{"role": "user", "content": prompt}]

        if history:
            messages = history + messages

        if isinstance(system_message, str):
            messages = [{"role": "system", "content": system_message}, *messages]
        elif isinstance(system_message, list):
            messages = [*system_message, *messages]

        return {
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

    def _complete(self, data: dict) -> dict:
        try:
            response = self.client.chat(model=self.model, stream=False, **data)
        except Exception as e:
            msg = f"Ollama completion failed: {str(e)}"
            raise RuntimeError(msg) from e

        return {
            "completion": response.message.content,
            "tokens_prompt": response.prompt_eval_count,
            "tokens_completion": response.eval_count,
        }

    async def _acomplete(self, data: dict) -> dict:
        try:
            response = await self.async_client.chat(model=self.model, stream=False, **data)
        except Exception as e:
            msg = f"Ollama completion failed: {str(e)}"
            raise RuntimeError(msg) from e

        return {
            "completion": response.message.content,
            "tokens_prompt": response.prompt_eval_count,
            "tokens_completion": response.eval_count,
        }

    def _complete_stream(self, data: dict) -> t.Iterator[str]:
        for chunk in self.client.chat(model=self.model, stream=True, **data):
            if c := chunk["message"]["content"]:
                yield c

    async def _acomplete_stream(self, data: dict) -> t.AsyncIterator[str]:
        async for chunk in await self.async_client.chat(model=self.model, stream=True, **data):
            if c := chunk["message"]["content"]:
                yield c
