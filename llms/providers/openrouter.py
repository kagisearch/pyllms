from typing import AsyncGenerator, Dict, List, Optional, Union
import tiktoken

from openai import AsyncOpenAI, OpenAI

from ..results.result import AsyncStreamResult, Result, StreamResult
from .base_provider import BaseProvider


class OpenRouterProvider(BaseProvider):
    MODEL_INFO = {
        "nvidia/llama-3.1-nemotron-70b-instruct": {"prompt": 0.35, "completion": 0.4, "token_limit": 131072, "is_chat": True},
        "x-ai/grok-2": {"prompt": 5.0, "completion": 10.0, "token_limit": 32768, "is_chat": True},
        "nousresearch/hermes-3-llama-3.1-405b:free": {"prompt": 0.0, "completion": 0.0, "token_limit": 8192, "is_chat": True},
        "google/gemini-flash-1.5-exp": {"prompt": 0.0, "completion": 0.0, "token_limit": 1000000, "is_chat": True},
        "liquid/lfm-40b": {"prompt": 0.0, "completion": 0.0, "token_limit": 32768, "is_chat": True},
        "mistralai/ministral-8b": {"prompt": 0.1, "completion": 0.1, "token_limit": 128000, "is_chat": True},
        "qwen/qwen-2.5-72b-instruct": {"prompt": 0.35, "completion": 0.4, "token_limit": 131072, "is_chat": True},
        "openai/o1": {"prompt": 15.0, "completion": 60.0, "token_limit": 200000, "is_chat": True},
        "google/gemini-2.0-flash-thinking-exp:free": {"prompt": 0.0, "completion": 0.0, "token_limit": 40000, "is_chat": True},
        "x-ai/grok-2-1212": {"prompt": 2.0, "completion": 10.0, "token_limit": 131072, "is_chat": True},
        "google/gemini-exp-1206:free": {"prompt": 0.0, "completion": 0.0, "token_limit": 2100000, "is_chat": True},
        "google/gemini-2.0-flash-exp:free": {"prompt": 0.0, "completion": 0.0, "token_limit": 1050000, "is_chat": True},
        "deepseek/deepseek-r1-distill-llama-70b": {"prompt": 0.23, "completion": 0.69, "token_limit": 131000, "is_chat": True},
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
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", **client_kwargs)
        if async_client_kwargs is None:
            async_client_kwargs = {}
        self.async_client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", **async_client_kwargs)

    @property
    def is_chat_model(self) -> bool:
        return self.MODEL_INFO[self.model]['is_chat']

    def count_tokens(self, content: Union[str, List[dict]]) -> int:
        # OpenRouter uses the same tokenizer as OpenAI
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
            "extra_headers": {
                "HTTP-Referer": kwargs.get("site_url", ""),
                "X-Title": kwargs.get("app_name", ""),
            },
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

        if not response or not hasattr(response, 'choices') or not response.choices:
            raise ValueError("Unexpected response structure from OpenRouter API")

        completion = response.choices[0].message.content.strip() if response.choices[0].message else ""
        usage = response.usage if hasattr(response, 'usage') else None

        meta = {
            "tokens_prompt": usage.prompt_tokens if usage else 0,
            "tokens_completion": usage.completion_tokens if usage else 0,
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
