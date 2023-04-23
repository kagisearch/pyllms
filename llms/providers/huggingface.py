# llms/providers/cohere.py

import itertools
import os
from huggingface_hub.inference_api import InferenceApi
import time

from typing import List, Optional


class HuggingfaceHubProvider:
    MODEL_INFO = {
        "hf_pythia": {
            "full": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
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
            model = list(MODEL_INFO.keys())[0]

        self.model = model

        if api_key is None:
            api_key = os.getenv("HUGGINFACEHUB_API_KEY")

        self.client = InferenceApi(
            repo_id=self.MODEL_INFO[model]["full"], token=api_key
        )

    def __str__(self):
        return f"{self.__class__.__name__} ({self.model})"

    def complete(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 300,
        **kwargs,
    ):
        if self.model == "hf_pythia":
            prompt = "<|prompter|" + prompt + "<|endoftext|><|assistant|>"

        if temperature <=0:
            temperature = 0.01

        if "temperature" not in kwargs:
            kwargs["temperature"] = temperature

        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = max_tokens

        start_time = time.time()
        response = self.client(inputs=prompt, params={**kwargs})
        latency = time.time() - start_time
        #print(response)
        if 'error' in response:
            print("Error: ", response['error'])
            return {}


        completion = response[0]["generated_text"][len(prompt) :]

        # Calculate tokens and cost
        prompt_tokens = -1
        completion_tokens = -1

        total_tokens = prompt_tokens + completion_tokens
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
                "latency": latency,
            },
        }
