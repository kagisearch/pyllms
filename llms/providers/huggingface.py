from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import InferenceClient

from .base import ModelInfo, SyncProvider


@dataclass
class HuggingfaceHubProvider(SyncProvider):
    MODEL_INFO = {
        "hf_pythia": ModelInfo(
            hf_repo="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
        ),
        "hf_falcon40b": ModelInfo(
            hf_repo="tiiuae/falcon-40b-instruct",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_falcon7b": ModelInfo(
            hf_repo="tiiuae/falcon-7b-instruct",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_mptinstruct": ModelInfo(
            hf_repo="mosaicml/mpt-7b-instruct",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_mptchat": ModelInfo(
            hf_repo="mosaicml/mpt-7b-chat",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_llava": ModelInfo(
            hf_repo="liuhaotian/LLaVA-Lightning-MPT-7B-preview",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_dolly": ModelInfo(
            hf_repo="databricks/dolly-v2-12b",
            prompt_cost=0,
            completion_cost=0,
            context_limit=-1,
            local=True,
        ),
        "hf_vicuna": ModelInfo(
            hf_repo="CarperAI/stable-vicuna-13b-delta",
            prompt_cost=0,
            completion_cost=0,
            context_limit=-1,
        ),
    }

    def __post_init__(self, api_key=None, model=None):
        super().__post_init__()
        self.model = model
        self.client = InferenceClient(self.info.hf_repo, token=api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        raise

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.chat_completion(messages=messages, **kwargs)
        return {
            "completion": response.choices[0].message,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
