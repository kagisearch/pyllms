# https://developers.generativeai.google/api/python/google/generativeai


import os, math
from typing import Dict

import google.generativeai as genai

from ..results.result import Result
from .base_provider import BaseProvider


class GoogleGenAIProvider(BaseProvider):
    # cost is per million tokens
    MODEL_INFO = {
        # no support for "textembedding-gecko"
        "chat-bison-genai": {"prompt": 0.5, "completion": 0.5, "token_limit": 0, "uses_characters": True},
        "text-bison-genai": {"prompt": 1.0, "completion": 1.0, "token_limit": 0, "uses_characters": True},
        "gemini-1.5-pro": {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        "gemini-1.5-pro-latest": {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.3, "token_limit": 128000, "uses_characters": True},
        "gemini-1.5-flash-latest": {"prompt": 0.075, "completion": 0.3, "token_limit": 128000, "uses_characters": True},
        "gemini-1.5-pro-exp-0801" : {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        "gemini-2.0-flash-exp" : {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        "gemini-2.0-flash" : {"prompt": 0.1, "completion": 0.4, "token_limit": 128000, "uses_characters": True},
        "gemini-2.0-flash-lite-preview-02-05" : {"prompt": 0.075, "completion": 0.30, "token_limit": 128000, "uses_characters": True},
        "gemini-2.0-pro-exp-02-05" : {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        "gemini-2.5-pro-exp-03-25" : {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        "gemini-2.0-flash-thinking-exp-01-21" : {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        "gemini-2.5-flash-preview-04-17" : {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        "gemini-exp-1206" : {"prompt": 3.5, "completion": 10.5, "token_limit": 128000, "uses_characters": True},
        
    }
    
    def __init__(self, api_key=None, model=None, **kwargs):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]

        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        self.client = genai.configure(api_key=api_key)

        self.model = model
        if model.startswith('text-'):
            self.client = genai.generate_text
            self.mode = 'text'
        else:
            self.client = genai.GenerativeModel(model)
            self.mode = 'chat'


    def _prepare_model_inputs(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ) -> Dict:
        temperature = max(temperature, 0.01)
        max_output_tokens = kwargs.pop("max_output_tokens", max_tokens)
        if self.mode == 'chat':
            messages=kwargs.pop("messages", [])
            messages=messages + [prompt]
            model_inputs = {
                #"messages": messages,
                #"temperature": temperature,
                **kwargs,
            }
        else:
            model_inputs = {
                "prompt": prompt,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens, **kwargs,
            }
        return model_inputs

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
            response = self.client.generate_content(prompt)
        
        if self.mode == 'chat':
            completion = response.text
        else:
            completion = response.result

        if completion is None:
            completion=""
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
