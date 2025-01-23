from __future__ import annotations

import os
from dataclasses import dataclass

from anthropic import AnthropicBedrock, AsyncAnthropicBedrock

from .anthropic import AnthropicProvider


@dataclass
class BedrockAnthropicProvider(AnthropicProvider):
    api_key = ""
    MODEL_INFO = {
        "anthropic.claude-v2": {
            "prompt": 11.02,
            "completion": 32.68,
            "token_limit": 100_000,
        },
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "prompt": 0.25,
            "completion": 1.25,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "prompt": 3.00,
            "completion": 15,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "prompt": 3.00,
            "completion": 15,
            "token_limit": 200_000,
            "output_limit": 4_096,
        },
    }

    aws_access_key: str | None = None
    aws_secret_key: str | None = None
    aws_region: str | None = None

    def __post_init__(self):
        super().__post_init__()
        aws_access_key = self.aws_access_key or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = self.aws_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        self.client = AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=self.aws_region,
        )

        self.async_client = AsyncAnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=self.aws_region,
        )
