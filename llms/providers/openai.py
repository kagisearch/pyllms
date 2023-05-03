import aiohttp
import openai
import tiktoken
from typing import List, Optional

from .base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    # cost is per million tokens
    MODEL_INFO = {
        "gpt-3.5-turbo": {"prompt": 2.0, "completion": 2.0, "token_limit": 4000},
        "gpt-4": {"prompt": 30.0, "completion": 60.0, "token_limit": 8000},
    }

    def __init__(self, api_key, model=None):
        openai.api_key = api_key
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model

    def count_tokens(self, content: str):
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(content))

    def _prepapre_model_input(self,
                              prompt: str,
                              history: Optional[List[dict]] = None,
                              system_message: Optional[List[dict]] = None,
                              temperature: float = 0,
                              max_tokens: int = 300,
                              stream: bool = False,
                              **kwargs,
                              ):

        messages = [{"role": "user", "content": prompt}]

        if history:
            messages = [*history, *messages]

        if system_message:
            messages = [*system_message, *messages]

        model_input = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        return model_input

    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """

        model_input = self._prepapre_model_input(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        with self.track_latency() as latency:
            response = openai.ChatCompletion.create(model=self.model, **model_input)

        completion = response.choices[0].message.content.strip()
        usage = response.usage
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]

        cost = self.compute_cost(prompt_tokens=prompt_tokens,
                                 completion_tokens=completion_tokens
                                 )

        return {
            "text": completion,
            "meta": {
                "model": self.model,
                "tokens": total_tokens,
                "tokens_prompt": prompt_tokens,  # Add tokens_prompt to meta
                "tokens_completion": completion_tokens,  # Add tokens_completion to meta
                "cost": cost,
                "latency": latency,
            },
        }

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        aiosession: Optional[aiohttp.ClientSession] = None,
        **kwargs,
    ):
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """
        if aiosession is not None:
            openai.aiosession.set(aiosession)

        model_input = self._prepapre_model_input(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        with self.track_latency() as latency:
            response = await openai.ChatCompletion.acreate(model=self.model, **model_input)

        completion = response.choices[0].message.content.strip()
        usage = response.usage
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]

        cost = self.compute_cost(prompt_tokens=prompt_tokens,
                                 completion_tokens=completion_tokens
                                 )

        return {
            "text": completion,
            "meta": {
                "model": self.model,
                "tokens": total_tokens,
                "tokens_prompt": prompt_tokens,  # Add tokens_prompt to meta
                "tokens_completion": completion_tokens,  # Add tokens_completion to meta
                "cost": cost,
                "latency": latency,
            },
        }

    def complete_stream(
        self,
        prompt: str,
        history: Optional[List[tuple]] = None,
        system_message: str = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ):
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """
        model_input = self._prepapre_model_input(prompt=prompt,
                                                 history=history,
                                                 system_message=system_message,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens,
                                                 stream=True,
                                                 **kwargs
                                                 )
        response = openai.ChatCompletion.create(model=self.model, **model_input)

        chunk_generator = (
            chunk["choices"][0].get("delta", {}).get("content") for chunk in response
        )
        while not (first_text := next(chunk_generator)):
            continue
        yield first_text.lstrip()
        yield from chunk_generator
