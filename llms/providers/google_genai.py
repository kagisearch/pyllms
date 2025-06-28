# https://googleapis.github.io/python-genai/

import os, math
from typing import Dict, Generator, AsyncGenerator

from google import genai
from google.genai import types

from ..results.result import Result, StreamResult, AsyncStreamResult
from .base_provider import BaseProvider


class GoogleGenAIProvider(BaseProvider):
    # cost is per million tokens
    MODEL_INFO = {
        # Gemini 2.5 family - Enhanced thinking and reasoning
        "gemini-2.5-pro": {"prompt": 5.0, "completion": 15.0, "token_limit": 2000000, "uses_characters": True},
        "gemini-2.5-flash": {"prompt": 0.1, "completion": 0.4, "token_limit": 2000000, "uses_characters": True},
        "gemini-2.5-flash-lite-preview-06-17": {"prompt": 0.05, "completion": 0.2, "token_limit": 2000000, "uses_characters": True},
        
        # Gemini 2.0 family - Next generation features and speed
        "gemini-2.0-flash": {"prompt": 0.075, "completion": 0.3, "token_limit": 2000000, "uses_characters": True},
        "gemini-2.0-flash-lite": {"prompt": 0.0375, "completion": 0.15, "token_limit": 1000000, "uses_characters": True},
        
        # Gemini 1.5 family - Stable and reliable models
        "gemini-1.5-pro": {"prompt": 3.5, "completion": 10.5, "token_limit": 2000000, "uses_characters": True},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.3, "token_limit": 1000000, "uses_characters": True},
        "gemini-1.5-flash-8b": {"prompt": 0.0375, "completion": 0.15, "token_limit": 1000000, "uses_characters": True},
    }
    
    def __init__(self, api_key=None, model=None, **kwargs):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]

        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def count_tokens(self, content):
        """
        Count tokens in the given content.
        For Google GenAI, we'll use a simple approximation since the exact 
        token counting API might not be readily available in streaming context.
        """
        if isinstance(content, str):
            # Simple approximation: ~4 characters per token for most languages
            return max(1, len(content) // 4)
        elif isinstance(content, (list, dict)):
            # Convert to string and count
            import json
            content_str = json.dumps(content) if isinstance(content, dict) else str(content)
            return max(1, len(content_str) // 4)
        else:
            return max(1, len(str(content)) // 4)

    def _prepare_model_inputs(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        stream: bool = False,
        **kwargs,
    ) -> Dict:
        temperature = max(temperature, 0.01)
        max_output_tokens = kwargs.pop("max_output_tokens", max_tokens)
        
        # Create config using the modern API
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        return {"config": config, "contents": prompt}

    def complete(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        context: str = None,
        examples: dict = {},
        **kwargs,
    ) -> Result:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        with self.track_latency():
            response = self.client.models.generate_content(
                model=self.model,
                contents=model_inputs["contents"],
                config=model_inputs["config"],
            )
        
        completion = response.text or ""

        # Calculate tokens and cost
        prompt_tokens = len(prompt)
        completion_tokens = len(completion)

        cost_per_token = self.MODEL_INFO[self.model]
        cost = (
            (prompt_tokens * cost_per_token["prompt"])
            + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000

        # fast approximation. We could call count_message_tokens() but this will add latency
        prompt_tokens = math.ceil((prompt_tokens+1) / 4)
        completion_tokens = math.ceil((completion_tokens+1) / 4)
        total_tokens = math.ceil(prompt_tokens + completion_tokens)

        meta = {
            "model": self.model,
            "tokens": total_tokens,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
            "cost": cost,
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
        temperature: float = 0.01,
        max_tokens: int = 300,
        context: str = None,
        examples: dict = {},
        **kwargs,
    ) -> StreamResult:
        """
        Stream completion for Google GenAI provider.
        
        Args:
            prompt: The text prompt to complete
            temperature: Controls randomness (min 0.01 for Google)
            max_tokens: Maximum tokens to generate
            context: Additional context (unused in this implementation)
            examples: Examples dict (unused in this implementation)
            **kwargs: Additional parameters passed to the model
        """
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        
        with self.track_latency():
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=model_inputs["contents"],
                config=model_inputs["config"],
            )
            stream = self._process_stream(response)

        return StreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    def _process_stream(self, response) -> Generator:
        """
        Process the streaming response from Google GenAI.
        
        Args:
            response: The streaming response from Google's generate_content_stream
        
        Yields:
            str: Individual text chunks from the stream
        """
        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def acomplete_stream(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        context: str = None,
        examples: dict = {},
        **kwargs,
    ) -> AsyncStreamResult:
        """
        Async stream completion for Google GenAI provider.
        
        Args:
            prompt: The text prompt to complete
            temperature: Controls randomness (min 0.01 for Google)
            max_tokens: Maximum tokens to generate
            context: Additional context (unused in this implementation)
            examples: Examples dict (unused in this implementation)
            **kwargs: Additional parameters passed to the model
        """
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        
        with self.track_latency():
            response = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=model_inputs["contents"],
                config=model_inputs["config"],
            )
            stream = self._aprocess_stream(response)

        return AsyncStreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    async def _aprocess_stream(self, response) -> AsyncGenerator:
        """
        Process the async streaming response from Google GenAI.
        
        Args:
            response: The async streaming response from Google's generate_content_stream
        
        Yields:
            str: Individual text chunks from the stream
        """
        async for chunk in response:
            if chunk.text:
                yield chunk.text
