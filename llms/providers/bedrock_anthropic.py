# llms/providers/bedrock_anthropic.py

import os
from typing import Optional

import anthropic_bedrock

from .anthropic import AnthropicProvider

class BedrockAnthropicProvider(AnthropicProvider):
    MODEL_INFO = {
        "anthropic.claude-instant-v1": {"prompt": 1.63, "completion": 5.51, "token_limit": 9000},
        "anthropic.claude-v1": {
            "prompt": 11.02,
            "completion": 32.68,
            "token_limit": 100_000,
        },
        "anthropic.claude-v2": {"prompt": 11.02, "completion": 32.68, "token_limit": 100_000},
    }

    def __init__(
        self, 
        model: Optional[str] = None, 
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_region: Optional[str] = None
    ):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model

        if aws_access_key is None:
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        if aws_secret_key is None:
            aws_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.client = anthropic_bedrock.AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region
        )
        self.async_client = anthropic_bedrock.AsyncAnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region
        )

