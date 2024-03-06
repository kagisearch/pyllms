from typing import AsyncGenerator, Dict, Generator, List, Optional, Union
import tiktoken

from openai import AsyncOpenAI, OpenAI
import json

from ..results.result import AsyncStreamResult, Result, StreamResult
from .base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    # cost is per million tokens
    MODEL_INFO = {
        "gpt-3.5-turbo": {"prompt": 2.0, "completion": 2.0, "token_limit": 16_385, "is_chat": True, "output_limit": 4_096},
        "gpt-3.5-turbo-1106": {"prompt": 2.0, "completion": 2.0, "token_limit": 16_385, "is_chat": True, "output_limit": 4_096},
        "gpt-3.5-turbo-instruct": {"prompt": 2.0, "completion": 2.0, "token_limit": 4096, "is_chat": False},
        "gpt-4": {"prompt": 30.0, "completion": 60.0, "token_limit": 8192, "is_chat": True},
        "gpt-4-1106-preview": {"prompt": 10.0, "completion": 30.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4-turbo-preview": {"prompt": 10.0, "completion": 30.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
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
        self.client = OpenAI(api_key=api_key, **client_kwargs)
        if async_client_kwargs is None:
            async_client_kwargs = {}
        self.async_client = AsyncOpenAI(api_key=api_key, **async_client_kwargs)

    @property
    def is_chat_model(self) -> bool:
        return self.MODEL_INFO[self.model]['is_chat']

    def count_tokens(self, content: Union[str, List[dict]]) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        if isinstance(content, list):
            # When field name is present, ChatGPT will ignore the role token.
            # Adopted from OpenAI cookbook
            # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
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
        if self.is_chat_model:
            messages = [{"role": "user", "content": prompt}]

            if history:
                messages = [*history, *messages]

            if isinstance(system_message, str):
                messages = [{"role": "system", "content": system_message}, *messages]

            # users can input multiple full system message in dict form
            elif isinstance(system_message, list):
                messages = [*system_message, *messages]

            model_inputs = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
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
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        with self.track_latency():
            if self.is_chat_model:
                response = self.client.chat.completions.create(model=self.model, **model_inputs)
            else:
                response = self.client.completions.create(model=self.model, **model_inputs)

        is_func_call = response.choices[0].finish_reason == "function_call"
        function_call = {}
        completion = ""
        if self.is_chat_model:
            if is_func_call:
                function_call = {
                    "name": response.choices[0].message.function_call.name,
                    "arguments": json.loads(response.choices[0].message.function_call.arguments)
                }
            else:
                completion = response.choices[0].message.content.strip()
        else:
            completion = response.choices[0].text.strip()

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
            function_call=function_call,
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
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        with self.track_latency():
            if self.is_chat_model:
                response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)
            else:
                response = await self.async_client.completions.create(model=self.model, **model_inputs)

        if self.is_chat_model:
            completion = response.choices[0].message.content.strip()
        else:
            completion = response.choices[0].text.strip()

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
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        if self.is_chat_model:
            response = self.client.chat.completions.create(model=self.model, **model_inputs)
        else:
            response = self.client.completions.create(model=self.model, **model_inputs)
        stream = self._process_stream(response)

        return StreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    def _process_stream(self, response: Generator) -> Generator:
        if self.is_chat_model:
            chunk_generator = (
                chunk.choices[0].delta.content for chunk in response
            )
        else:
            chunk_generator = (
                chunk.choices[0].text for chunk in response
            )

        while not (first_text := next(chunk_generator)):
            continue
        yield first_text.lstrip()
        for chunk in chunk_generator:
            if chunk is not None:
                yield chunk

    async def acomplete_stream(
            self,
            prompt: str,
            history: Optional[List[dict]] = None,
            system_message: Union[str, List[dict], None] = None,
            temperature: float = 0,
            max_tokens: int = 300,
            **kwargs,
    ) -> AsyncStreamResult:
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        with self.track_latency():
            if self.is_chat_model:
                response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)
            else:
                response = await self.async_client.completions.create(model=self.model, **model_inputs)
        stream = self._aprocess_stream(response)
        return AsyncStreamResult(
            stream=stream, model_inputs=model_inputs, provider=self
        )

    async def _aprocess_stream(self, response: AsyncGenerator) -> AsyncGenerator:
        if self.is_chat_model:
            while True:
                first_completion = (await response.__anext__()).choices[0].delta.content
                if first_completion:
                    yield first_completion.lstrip()
                    break

            async for chunk in response:
                completion = chunk.choices[0].delta.content
                if completion is not None:
                    yield completion
        else:
            while True:
                first_completion = (await response.__anext__()).choices[0].text
                if first_completion:
                    yield first_completion.lstrip()
                    break

            async for chunk in response:
                completion = chunk.choices[0].text
                if completion is not None:
                    yield completion
