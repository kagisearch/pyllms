# llms/providers/huggingface.py

import os

from huggingface_hub.inference_api import InferenceApi

from .base_provider import BaseProvider


class HuggingfaceHubProvider(BaseProvider):
    MODEL_INFO = {
        "hf_pythia": {
            "full": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
        },
        "hf_mptinstruct": {
            "full": "mosaicml/mpt-7b-instruct",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
        },
        "hf_mptchat": {
            "full": "mosaicml/mpt-7b-chat",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
        },
        "hf_llava": {
            "full": "liuhaotian/LLaVA-Lightning-MPT-7B-preview",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
        },
        "hf_dolly": {
            "full": "databricks/dolly-v2-12b",
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

    def _prepare_model_input(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ):
        if self.model == "hf_pythia":
            prompt = "<|prompter|" + prompt + "<|endoftext|><|assistant|>"
        temperature = max(temperature, 0.01)
        max_new_tokens = kwargs.pop("max_new_tokens", max_tokens)
        params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }
        return prompt, params

    def complete(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ):
        prompt, params = self._prepare_model_input(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        with self.track_latency():
            response = self.client(inputs=prompt, params=params)

        if "error" in response:
            print("Error: ", response["error"])
            return {}

        completion = response[0]["generated_text"][len(prompt) :]

        # Calculate tokens and cost
        prompt_tokens = -1
        completion_tokens = -1

        total_tokens = prompt_tokens + completion_tokens

        cost = self.compute_cost(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        cost_per_token = self.MODEL_INFO[self.model]
        cost = (
            (prompt_tokens * cost_per_token["prompt"])
            + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000

        return {
            "text": completion,
            "meta": {
                "model": self.model,
                "tokens": total_tokens,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
                "cost": cost,
                "latency": self.latency,
            },
            "provider": str(self),
        }
