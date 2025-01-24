from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import InferenceClient

from .base import SyncProvider


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

    def _count_tokens(self, content: list[dict]) -> int:
        raise

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.chat_completion(messages=messages, **kwargs)
        return {
            "completion": response.choices[0].message,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
