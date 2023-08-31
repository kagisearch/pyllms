# we could switch to genai  https://developers.generativeai.google/api/python/google/generativeai


from typing import Dict

import math
import vertexai
from vertexai.language_models import TextGenerationModel, ChatModel, CodeGenerationModel, CodeChatModel, InputOutputTextPair

from ..results.result import Result
from .base_provider import BaseProvider


class GoogleProvider(BaseProvider):
    # cost is per million tokens
    MODEL_INFO = {
        # no support for "textembedding-gecko"
        "chat-bison": {"prompt": 0.5, "completion": 0.5, "token_limit": 0, "uses_characters": True},
        "text-bison": {"prompt": 1.0, "completion": 1.0, "token_limit": 0, "uses_characters": True},
        "text-bison-32k": {"prompt": 1.0, "completion": 1.0, "token_limit": 0, "uses_characters": True},
        "code-bison": {"prompt": 1.0, "completion": 1.0, "token_limit": 0, "uses_characters": True},
        "code-bison-32k": {"prompt": 1.0, "completion": 1.0, "token_limit": 0, "uses_characters": True},
        "codechat-bison": {"prompt": 1.0, "completion": 1.0, "token_limit": 0, "uses_characters": True},    
        "codechat-bison-32k": {"prompt": 1.0, "completion": 1.0, "token_limit": 0, "uses_characters": True},    
     }
    
    def __init__(self, model=None, **kwargs):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]

        self.model = model
        if model.startswith('text-'):
            self.client = TextGenerationModel.from_pretrained(model)
            self.prompt_key = 'prompt'
        elif model.startswith('code-'):
            self.client = CodeGenerationModel.from_pretrained(model)
            self.prompt_key = 'prefix'
        elif model.startswith('codechat-'):
            self.client = CodeChatModel.from_pretrained(model)
            self.prompt_key = 'message'
        else:
            self.client = ChatModel.from_pretrained(model)
            self.prompt_key = 'message'
        
        vertexai.init(**kwargs)

    def _prepare_model_inputs(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ) -> Dict:
        temperature = max(temperature, 0.01)
        max_output_tokens = kwargs.pop("max_output_tokens", max_tokens)
        model_inputs = {
            self.prompt_key: prompt,
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
            if isinstance(self.client, ChatModel) or isinstance(self.client, CodeChatModel):
                chat = self.client.start_chat()
                response = chat.send_message(**model_inputs)
            else: # text / code
                response = self.client.predict(**model_inputs)
        
        completion = response.text

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

        prompt_tokens = math.ceil((prompt_tokens+1) / 4)
        completion_tokens = math.ceil((completion_tokens+1) / 4)
        total_tokens = prompt_tokens + completion_tokens

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
