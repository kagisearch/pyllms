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
        "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4o": {"prompt": 2.5, "completion": 10.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4o-2024-08-06": {"prompt": 2.50, "completion": 10.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4.1": {"prompt": 10.0, "completion": 30.0, "token_limit": 128000, "is_chat": True, "output_limit": 16384},
        "gpt-4.1-mini": {"prompt": 2.0, "completion": 8.0, "token_limit": 128000, "is_chat": True, "output_limit": 16384},
        "gpt-4.1-nano": {"prompt": 0.5, "completion": 2.0, "token_limit": 128000, "is_chat": True, "output_limit": 16384},
        "gpt-4.5-preview": {"prompt": 75, "completion": 150.0, "token_limit": 128000, "is_chat": True, "output_limit": 16384},
        "chatgpt-4o-latest": {"prompt": 5, "completion": 15.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "o1-preview": {"prompt": 15.0, "completion": 60.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096, "use_max_completion_tokens": True},
        "o1-mini": {"prompt": 3.0, "completion": 12.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096, "use_max_completion_tokens": True},
        "o1": {"prompt": 15.0, "completion": 60.0, "token_limit": 200000, "is_chat": True, "output_limit": 100000, "use_max_completion_tokens": True},
        "o1-pro": {"prompt": 150.0, "completion": 600.0, "token_limit": 200000, "is_chat": True, "output_limit": 100000, "use_max_completion_tokens": True, "use_responses_api": True},
        "o3-mini": {"prompt": 1.1, "completion": 4.40, "token_limit": 128000, "is_chat": True, "output_limit": 4_096, "use_max_completion_tokens": True},
        "o3": {"prompt": 20.0, "completion": 80.0, "token_limit": 200000, "is_chat": True, "output_limit": 100000, "use_max_completion_tokens": True},
        "o3-pro": {"prompt": 200.0, "completion": 800.0, "token_limit": 200000, "is_chat": True, "output_limit": 100000, "use_max_completion_tokens": True, "use_responses_api": True},
        "o4-mini": {"prompt": 0.8, "completion": 3.2, "token_limit": 128000, "is_chat": True, "output_limit": 4_096, "use_max_completion_tokens": True},
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
        
    @property
    def uses_responses_api(self) -> bool:
        return self.MODEL_INFO[self.model].get('use_responses_api', False)

    def count_tokens(self, content: Union[str, List[dict]]) -> int:
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # For new models not yet in tiktoken, use gpt-4 as fallback
            enc = tiktoken.encoding_for_model("gpt-4")
        
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
        reasoning_effort: Optional[str] = None,
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
                "stream": stream,
                **({'reasoning_effort': reasoning_effort} if reasoning_effort else {}),
                **kwargs,
            }

            # Use max_completion_tokens for models that require it
            if self.MODEL_INFO[self.model].get("use_max_completion_tokens", False):
                model_inputs["max_completion_tokens"] = max_tokens
            else:
                model_inputs["max_tokens"] = max_tokens
                model_inputs["temperature"] = temperature

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
            if self.uses_responses_api:
                # Convert messages format for Responses API
                input_messages = model_inputs.pop("messages")
                # Handle any reasoning_effort parameter
                reasoning = {}
                if "reasoning_effort" in model_inputs:
                    reasoning["effort"] = model_inputs.pop("reasoning_effort")
                
                # Prepare parameters for Responses API
                responses_params = {
                    "model": self.model,
                    "input": input_messages
                }
                
                # Temperature is not supported for some models with Responses API
                # Only add it if the model supports it
                
                # For Responses API, max_tokens should be converted to max_output_tokens
                if max_tokens is not None:
                    responses_params["max_output_tokens"] = max_tokens
                
                # Add any other supported parameters
                for key, value in model_inputs.items():
                    if key not in ["messages", "max_completion_tokens", "max_tokens", "temperature", "reasoning_effort"]:
                        responses_params[key] = value
                
                # Add reasoning if present
                if reasoning:
                    responses_params["reasoning"] = reasoning
                
                response = self.client.responses.create(**responses_params)
            elif self.is_chat_model:
                response = self.client.chat.completions.create(model=self.model, **model_inputs)
            else:
                response = self.client.completions.create(model=self.model, **model_inputs)

        function_call = {}
        completion = ""
        
        if self.uses_responses_api:
            # Extract text from Responses API
            # Find the output_text in the response
            for item in response.output:
                if item.type == "message" and hasattr(item, "content"):
                    for content_item in item.content:
                        if content_item.type == "output_text":
                            completion = content_item.text.strip()
                            break
            
            # Handle function calls if present
            if hasattr(response, 'output') and hasattr(response.output, 'function_calls'):
                function_call = {
                    "name": response.output.function_calls[0].name,
                    "arguments": response.output.function_calls[0].arguments
                }
            
            # Usage has different field names in Responses API
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens
            }
        else:
            is_func_call = response.choices[0].finish_reason == "function_call"
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
            "tokens_prompt": usage["prompt_tokens"] if isinstance(usage, dict) else usage.prompt_tokens,
            "tokens_completion": usage["completion_tokens"] if isinstance(usage, dict) else usage.completion_tokens,
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
            if self.uses_responses_api:
                # Convert messages format for Responses API
                input_messages = model_inputs.pop("messages")
                # Handle any reasoning_effort parameter
                reasoning = {}
                if "reasoning_effort" in model_inputs:
                    reasoning["effort"] = model_inputs.pop("reasoning_effort")
                
                # Prepare parameters for Responses API
                responses_params = {
                    "model": self.model,
                    "input": input_messages
                }
                
                # Temperature is not supported for some models with Responses API
                # Only add it if the model supports it
                
                # For Responses API, max_tokens should be converted to max_output_tokens
                if max_tokens is not None:
                    responses_params["max_output_tokens"] = max_tokens
                
                # Add any other supported parameters
                for key, value in model_inputs.items():
                    if key not in ["messages", "max_completion_tokens", "max_tokens", "temperature", "reasoning_effort"]:
                        responses_params[key] = value
                
                # Add reasoning if present
                if reasoning:
                    responses_params["reasoning"] = reasoning
                
                response = await self.async_client.responses.create(**responses_params)
                # Find the output_text in the response
                completion = ""
                for item in response.output:
                    if item.type == "message" and hasattr(item, "content"):
                        for content_item in item.content:
                            if content_item.type == "output_text":
                                completion = content_item.text.strip()
                                break
                
                # Usage has different field names in Responses API
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            elif self.is_chat_model:
                response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)
                completion = response.choices[0].message.content.strip()
                usage = response.usage
            else:
                response = await self.async_client.completions.create(model=self.model, **model_inputs)
                completion = response.choices[0].text.strip()
                usage = response.usage

        # Handle usage consistently
        if isinstance(usage, dict):
            meta = {
                "tokens_prompt": usage["prompt_tokens"],
                "tokens_completion": usage["completion_tokens"],
                "latency": self.latency,
            }
        else:
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

        if self.uses_responses_api:
            # Responses API doesn't support streaming in the same way
            # For now, we'll use the chat completions API for streaming
            response = self.client.chat.completions.create(model=self.model, **model_inputs)
        elif self.is_chat_model:
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
            if self.uses_responses_api:
                # Responses API doesn't support streaming in the same way
                # For now, we'll use the chat completions API for streaming
                response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)
            elif self.is_chat_model:
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
