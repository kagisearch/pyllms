# llms/providers/huggingface.py

import os

from huggingface_hub.inference_api import InferenceApi

from ..results.result import Result
from .base_provider import BaseProvider


class HuggingfaceHubProvider(BaseProvider):
    MODEL_INFO = {
        "hf_pythia": {
            "full": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
        },
         "hf_falcon40b": {
            "full": "tiiuae/falcon-40b-instruct",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
            "local": True
        },
        "hf_falcon7b": {         
            "full": "tiiuae/falcon-7b-instruct",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
            "local": True
        },
        "hf_mptinstruct": {
            "full": "mosaicml/mpt-7b-instruct",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
            "local": True
        },
        "hf_mptchat": {
            "full": "mosaicml/mpt-7b-chat",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
           "local": True
        },
        "hf_llava": {
            "full": "liuhaotian/LLaVA-Lightning-MPT-7B-preview",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
            "local": True
        },
        "hf_dolly": {
            "full": "databricks/dolly-v2-12b",
            "prompt": 0,
            "completion": 0,
            "token_limit": -1,
            "local": True
        },
        "hf_vicuna": {
            "full": "CarperAI/stable-vicuna-13b-delta",
            "prompt": 0,
            "completion": 0,
            "token_limit": -1,
        },
    }

    def __init__(self, api_key=None, model=None):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]

        self.model = model

        if api_key is None:
            api_key = os.getenv("HUGGINFACEHUB_API_KEY")

        self.client = InferenceApi(
            repo_id=self.MODEL_INFO[model]["full"], token=api_key
        )

    def _prepare_model_inputs(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 300,
        **kwargs,
    ):
        if self.model == "hf_pythia":
            prompt = "<|prompter|" + prompt + "<|endoftext|><|assistant|>"
        max_new_tokens = kwargs.pop("max_length", max_tokens)
        params = {
            "temperature": temperature,
            "max_length": max_new_tokens,
            **kwargs,
        }
        return prompt, params

    def complete(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ) -> Result:
        prompt, params = self._prepare_model_inputs(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        with self.track_latency():
            response = self.client(inputs=prompt, params=params)

        completion = response[0]["generated_text"][len(prompt) :]
        meta = {
            "tokens_prompt": -1,
            "tokens_completion": -1,
            "latency": self.latency,
        }
        return Result(
            text=completion,
            model_inputs={"prompt": prompt, **params},
            provider=self,
            meta=meta,
        )
