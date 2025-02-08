from typing import AsyncGenerator, Dict, List, Optional, Union
import tiktoken

from openai import AsyncOpenAI, OpenAI

from ..results.result import AsyncStreamResult, Result, StreamResult
from .base_provider import BaseProvider


class GroqProvider(BaseProvider):
    MODEL_INFO = {
        "llama-3.1-405b-reasoning": {"prompt": 0.59, "completion": 0.79, "token_limit": 131072, "is_chat": True},
        "llama-3.1-70b-versatile": {"prompt": 0.59, "completion": 0.79, "token_limit": 131072, "is_chat": True},
        "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08, "token_limit": 131072, "is_chat": True},
        "gemma2-9b-it": {"prompt": 0.20, "completion": 0.20, "token_limit": 131072, "is_chat": True},
        "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79, "token_limit": 131072, "is_chat": True},
        "deepseek-r1-distill-llama-70b": {"prompt": 0.59, "completion": 0.99, "token_limit": 131072, "is_chat": True},
    }

    def __init__(
        self,
        api_key: Union[str, None] = None,
        model: Union[str, None] = None,
        client_kwargs: Union[dict, None] = None,
        async_client_kwargs: Union[dict, None] = None,
    ):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model
        if client_kwargs is None:
            client_kwargs = {}
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", **client_kwargs)
        if async_client_kwargs is None:
            async_client_kwargs = {}
        self.async_client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", **async_client_kwargs)

    @property
    def is_chat_model(self) -> bool:
        return self.MODEL_INFO[self.model]['is_chat']

    def count_tokens(self, content: Union[str, List[dict]]) -> int:
        # Groq uses the same tokenizer as OpenAI
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        if isinstance(content, list):
            formatting_token_count = 4
            messages = content
            messages_text = ["".join(message.values()) for message in messages]
            tokens = [enc.encode(t, disallowed_special=()) for t in messages_text]

            n_tokens_list = []
            for token, message in zip(tokens, messages):
                n_tokens = len(token) + formatting_token_count
                if "name" in message:
                    n_tokens += -1
                n_tokens_list.append(n_tokens)
            return sum(n_tokens_list)
        else:
            return len(enc.encode(content, disallowed_special=()))

    def _prepare_model_inputs(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Union[str, List[dict], None] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stream: bool = False,
        **kwargs,
    ) -> Dict:
        messages = [{"role": "user", "content": prompt}]

        if history:
            messages = [*history, *messages]

        if isinstance(system_message, str):
            messages = [{"role": "system", "content": system_message}, *messages]
        elif isinstance(system_message, list):
            messages = [*system_message, *messages]

        model_inputs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        return model_inputs

    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> Result:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        with self.track_latency():
            response = self.client.chat.completions.create(model=self.model, **model_inputs)

        completion = response.choices[0].message.content.strip()
        usage = response.usage

        meta = {
            "tokens_prompt": usage.prompt_tokens,
            "tokens_completion": usage.completion_tokens,
            "latency": self.latency,
        }
        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta=meta,
        )

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> Result:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        with self.track_latency():
            response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)

        completion = response.choices[0].message.content.strip()
        usage = response.usage

        meta = {
            "tokens_prompt": usage.prompt_tokens,
            "tokens_completion": usage.completion_tokens,
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
        system_message: Union[str, List[dict], None] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> StreamResult:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        response = self.client.chat.completions.create(model=self.model, **model_inputs)
        stream = self._process_stream(response)

        return StreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    def _process_stream(self, response):
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    async def acomplete_stream(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Union[str, List[dict], None] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> AsyncStreamResult:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)
        stream = self._aprocess_stream(response)
        return AsyncStreamResult(
            stream=stream, model_inputs=model_inputs, provider=self
        )

    async def _aprocess_stream(self, response):
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
