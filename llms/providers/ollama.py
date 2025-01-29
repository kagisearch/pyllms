import json
from typing import Any, Dict, Generator, List, Optional, Union, AsyncGenerator

from ollama import Client, AsyncClient

from ..results.result import Result, StreamResult, AsyncStreamResult
from .base_provider import BaseProvider


def _get_model_info(ollama_host: Optional[str] = "http://localhost:11434"):
    model_info = {}
    try:
        pulled_models = Client(host=ollama_host).list().get("models", [])
        for model in pulled_models:
            name = model["name"]
            # Ollama models are free to use locally
            model_info[name] = {
                "prompt": 0.0,
                "completion": 0.0,
                "token_limit": 4096  # Default token limit
            }

        if not pulled_models:
            raise ValueError("Could not retrieve any models from Ollama")
    except Exception as e:
        # Log the error but continue with empty model info
        pass
        #print(f"Warning: Could not connect to Ollama server: {str(e)}")
   
    return model_info


class OllamaProvider(BaseProvider):
    MODEL_INFO = _get_model_info()

    def count_tokens(self, content: Union[str, List[Dict[str, Any]]]) -> int:
        """Estimate token count using simple word-based heuristic"""
        if isinstance(content, list):
            # For chat messages, concatenate all content
            text = " ".join(msg["content"] for msg in content)
        else:
            text = content
        # Rough estimation: split on whitespace and punctuation
        return len(text.split())

    def __init__(
        self,
        model: Optional[str] = None,
        ollama_host: Optional[str] = "http://localhost:11434",
        ollama_client_options: Optional[dict] = None
    ):
        self.model = model
        if self.model is None:
            self.model = list(self.MODEL_INFO.keys())[0]

        if ollama_client_options is None:
            ollama_client_options = {}

        self.client = Client(host=ollama_host, **ollama_client_options)
        self.async_client = AsyncClient(host=ollama_host, **ollama_client_options)
        self.is_chat_model = True

    def _prepare_model_inputs(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Union[str, List[dict], None] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,  # Add but don't use
        temperature: Optional[float] = None,  # Add but don't use
        **kwargs
    ) -> Dict:
        # Remove unsupported parameters
        kwargs.pop('max_tokens', None)
        kwargs.pop('temperature', None)
        if self.is_chat_model:
            messages = [{"role": "user", "content": prompt}]

            if history:
                messages = history + messages


            if isinstance(system_message, str):
                messages = [{"role": "system", "content": system_message}, *messages]
            elif isinstance(system_message, list):
                messages = [*system_message, *messages]

            model_inputs = {
                "messages": messages,
                "stream": stream,
                **kwargs,
            }
        else:
            if history:
                raise ValueError(
                    f"history argument is not supported for {self.model} model"
                )

            if system_message:
                raise ValueError(
                    f"system_message argument is not supported for {self.model} model"
                )

            model_inputs = {
                "prompt": prompt,
                "stream": stream,
                **kwargs,
            }

        return model_inputs

    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        **kwargs
    ) -> Result:
        try:
            model_inputs = self._prepare_model_inputs(
                prompt=prompt,
                history=history,
                system_message=system_message,
                **kwargs
            )

            with self.track_latency():
                response = self.client.chat(model=self.model, **model_inputs)

                message = response["message"]
                completion = message["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama completion failed: {str(e)}")

        meta = {
            "tokens_prompt": response["prompt_eval_count"],
            "tokens_completion": response["eval_count"],
            "latency": self.latency,
        }

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta=meta,
        )

    def complete_stream(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        safe_prompt: bool = False,
        random_seed: Union[int, None] = None,
        **kwargs,
    ) -> StreamResult:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            stream=True,
            **kwargs
        )

        with self.track_latency():
            response = self.client.chat(model=self.model, **model_inputs)
            stream = self._process_stream(response=response)

            return StreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    def _process_stream(self, response: Generator) -> Generator:
        chunk_generator = (chunk["message"]["content"] for chunk in response)

        while not (first_text := next(chunk_generator)):
            continue

        yield first_text.lstrip()
        for chunk in chunk_generator:
            if chunk is not None:
                yield chunk

    async def _aprocess_stream(self, response) -> AsyncGenerator:
        while True:
            first_completion = (await response.__anext__())["message"]["content"]
            if first_completion:
                yield first_completion.lstrip()
                break

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        **kwargs
    ) -> Result:
        try:
            model_inputs = self._prepare_model_inputs(
                prompt=prompt,
                history=history,
                system_message=system_message,
                **kwargs
            )

            with self.track_latency():
                response = await self.async_client.chat(model=self.model, **model_inputs)

            message = response["message"]
            completion = ""
            completion = message["content"].strip()

            meta = {
                "tokens_prompt": response["prompt_eval_count"],
                "tokens_completion": response["eval_count"],
                "latency":  self.latency,
            }
        except Exception as e:
            raise RuntimeError(f"Ollama completion failed: {str(e)}")

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta=meta,
        )

    async def acomplete_stream(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        safe_prompt: bool = False,
        random_seed: Union[int, None] = None,
        **kwargs
    ):
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            stream=True,
            **kwargs
        )

        with self.track_latency():
            response = self.async_client.chat(model=self.model, **model_inputs)
            stream = self._aprocess_stream(response=response)

            return AsyncStreamResult(stream=stream, model_inputs=model_inputs, provider=self)
