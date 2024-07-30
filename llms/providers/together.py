from typing import AsyncGenerator, Dict, List, Optional, Union
import tiktoken

from together import Together

from ..results.result import AsyncStreamResult, Result, StreamResult
from .base_provider import BaseProvider


class TogetherProvider(BaseProvider):
    MODEL_INFO = {
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt": 5.0, "completion": 5.0, "token_limit": 4096},
    }

    def __init__(
        self,
        api_key: Union[str, None] = None,
        model: Union[str, None] = None,
        **kwargs
    ):
        super().__init__(api_key=api_key, model=model)
        if model is None:
            model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        self.model = model
        self.client = Together(api_key=api_key)

    def count_tokens(self, content: Union[str, List[dict]]) -> int:
        # Together uses the same tokenizer as OpenAI
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        if isinstance(content, list):
            return sum([len(enc.encode(str(message))) for message in content])
        else:
            return len(enc.encode(content))

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
        messages = [{"content": prompt, "role": "user"}]

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
        prompt_tokens = self.count_tokens(model_inputs["messages"])
        completion_tokens = self.count_tokens(completion)

        meta = {
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
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
            yield chunk.choices[0].delta.content

    # Note: Async methods are not implemented for Together AI as their Python SDK doesn't support async operations
