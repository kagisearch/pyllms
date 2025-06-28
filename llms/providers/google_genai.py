# https://googleapis.github.io/python-genai/

import os, math
from typing import Dict

from google import genai
from google.genai import types

from ..results.result import Result
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


    def _prepare_model_inputs(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
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
