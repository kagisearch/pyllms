from typing import AsyncGenerator, Dict, Generator, List, Optional

import aiohttp
import tiktoken

import openai
import json

from ..results.result import AsyncStreamResult, Result, StreamResult
from .base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    # cost is per million tokens
    MODEL_INFO = {

        "gpt-3.5-turbo": {"prompt": 2.0, "completion": 2.0, "token_limit": 8192,"is_chat": True},
        "gpt-3.5-turbo-instruct": {"prompt": 2.0, "completion": 2.0, "token_limit": 8192, "is_chat": False },
        "gpt-4": {"prompt": 30.0, "completion": 60.0, "token_limit": 8192,"is_chat": True},
       
    }

    def __init__(self, api_key, model=None):
        openai.api_key = api_key
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model
        self.client = openai.ChatCompletion if self.is_chat_model else openai.Completion

    @property
    def is_chat_model(self) -> bool:
        return self.MODEL_INFO[self.model]['is_chat']

    def count_tokens(self, content: str) -> int:
        # count tokens for chat model coming soon
        # this is mostly needed for stream result
        if self.is_chat_model:
            return 0
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(content))

    def _prepapre_model_inputs(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: str = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stream: bool = False,
        **kwargs,
    ) -> Dict:
        if self.is_chat_model:
            messages = [{"role": "user", "content": prompt}]

            if history:
                messages = [*history, *messages]

            if system_message:
                messages = [{"role": "system", "content": system_message}, *messages]

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
        
        model_inputs = self._prepapre_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        with self.track_latency():
            response = self.client.create(model=self.model, **model_inputs)
        
        is_func_call = response.choices[0].finish_reason == "function_call"
        if self.is_chat_model:
            if is_func_call:
                completion = {
                    "name": response.choices[0].message.function_call.name,
                    "arguments": json.loads(response.choices[0].message.function_call.arguments)
                }
            else:
                completion = response.choices[0].message.content.strip()
        else:
            completion = response.choices[0].text.strip()
        
        usage = response.usage
        
        meta = {
            "tokens_prompt": usage["prompt_tokens"],
            "tokens_completion": usage["completion_tokens"],
            "latency": self.latency,
        }
        return Result(
            text=completion if not is_func_call else '',
            model_inputs=model_inputs,
            provider=self,
            meta=meta,
            function_call=completion if is_func_call else {}
        )

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        aiosession: Optional[aiohttp.ClientSession] = None,
        **kwargs,
    ) -> Result:
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """
        if aiosession is not None:
            openai.aiosession.set(aiosession)

        model_inputs = self._prepapre_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        with self.track_latency():
            response = await self.client.acreate(model=self.model, **model_inputs)

        if self.is_chat_model:
            completion = response.choices[0].message.content.strip()
        else:
            completion = response.choices[0].text.strip()

        usage = response.usage

        meta = {
            "tokens_prompt": usage["prompt_tokens"],
            "tokens_completion": usage["completion_tokens"],
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
        history: Optional[List[tuple]] = None,
        system_message: str = None,
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
        model_inputs = self._prepapre_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        response = self.client.create(model=self.model, **model_inputs)
        stream = self._process_stream(response)

        return StreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    def _process_stream(self, response: Generator) -> Generator:
        if self.is_chat_model:
            chunk_generator = (
                chunk["choices"][0].get("delta", {}).get("content", "")
                for chunk in response
            )
        else:
            chunk_generator = (
                chunk["choices"][0].get("text", "") for chunk in response
            )

        while not (first_text := next(chunk_generator)):
            continue
        yield first_text.lstrip()
        yield from chunk_generator

    async def acomplete_stream(
            self,
            prompt: str,
            history: Optional[List[tuple]] = None,
            system_message: str = None,
            temperature: float = 0,
            max_tokens: int = 300,
            aiosession: Optional[aiohttp.ClientSession] = None,
            **kwargs,
    ) -> AsyncStreamResult:
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """
        if aiosession is not None:
            openai.aiosession.set(aiosession)

        model_inputs = self._prepapre_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        response = await self.client.acreate(model=self.model, **model_inputs)
        stream = self._aprocess_stream(response)
        return AsyncStreamResult(
            stream=stream, model_inputs=model_inputs, provider=self
        )

    async def _aprocess_stream(self, response: AsyncGenerator) -> AsyncGenerator:
        if self.is_chat_model:
            while True:
                first_completion = (await response.__anext__())["choices"][0].get("delta", {}).get("content", "")
                if first_completion:
                    yield first_completion.lstrip()
                    break
            
            async for chunk in response:
                completion = chunk["choices"][0].get("delta", {}).get("content", "")
                yield completion
        else:
            while True:
                first_completion = (await response.__anext__())["choices"][0].get("text", "")
                if first_completion:
                    yield first_completion.lstrip()
                    break
            
            async for chunk in response:
                completion = chunk["choices"][0].get("text", "")
                yield completion
