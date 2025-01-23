from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import InferenceClient

from .base import SyncProvider, from_raw


@dataclass
class HuggingfaceHubProvider(SyncProvider):
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
            "local": True,
        },
        "hf_falcon7b": {
            "full": "tiiuae/falcon-7b-instruct",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
            "local": True,
        },
        "hf_mptinstruct": {
            "full": "mosaicml/mpt-7b-instruct",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
            "local": True,
        },
        "hf_mptchat": {
            "full": "mosaicml/mpt-7b-chat",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
            "local": True,
        },
        "hf_llava": {
            "full": "liuhaotian/LLaVA-Lightning-MPT-7B-preview",
            "prompt": 0,
            "completion": 0,
            "token_limit": 2048,
            "local": True,
        },
        "hf_dolly": {
            "full": "databricks/dolly-v2-12b",
            "prompt": 0,
            "completion": 0,
            "token_limit": -1,
            "local": True,
        },
        "hf_vicuna": {
            "full": "CarperAI/stable-vicuna-13b-delta",
            "prompt": 0,
            "completion": 0,
            "token_limit": -1,
        },
    }

    def __post_init__(self, api_key=None, model=None):
        super().__post_init__()
        self.model = model
        self.client = InferenceClient(self.MODEL_INFO[model]["full"], token=api_key)

    def _prepare_input(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 300,
        **kwargs,
    ) -> dict:
        if self.model == "hf_pythia":
            prompt = "<|prompter|" + prompt + "<|endoftext|><|assistant|>"
        max_new_tokens = kwargs.pop("max_length", max_tokens)
        return {
            "prompt": from_raw(prompt),
            "temperature": temperature,
            "max_length": max_new_tokens,
            **kwargs,
        }

    def _count_tokens(self, content: list[dict]) -> int:
        raise

    def _complete(self, data: dict) -> dict:
        prompt: dict = data.pop("prompt")
        response = self.client.chat_completion(messages=[prompt], **data)
        return {
            "completion": response.choices[0].message,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
        }
