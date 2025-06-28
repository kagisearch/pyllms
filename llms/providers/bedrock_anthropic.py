# llms/providers/bedrock_anthropic.py

import os
from typing import Union, Dict
import tiktoken

from anthropic import AnthropicBedrock, AsyncAnthropicBedrock

from .anthropic import AnthropicProvider


class BedrockAnthropicProvider(AnthropicProvider):
    MODEL_INFO = {
        "anthropic.claude-3-haiku-20240307-v1:0": {"prompt": 0.25, "completion": 1.25, "token_limit": 200_000, "output_limit": 4_096},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"prompt": 3.00, "completion": 15, "token_limit": 200_000, "output_limit": 4_096},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt": 3.00, "completion": 15, "token_limit": 200_000, "output_limit": 4_096},
    }

    def __init__(
        self,
        model: Union[str, None] = None,
        aws_access_key: Union[str, None] = None,
        aws_secret_key: Union[str, None] = None,
        aws_region: Union[str, None] = None,
        client_kwargs: Union[dict, None] = None,
        async_client_kwargs: Union[dict, None] = None,
    ):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model

        if aws_access_key is None:
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        if aws_secret_key is None:
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if client_kwargs is None:
            client_kwargs = {}
        self.client = AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
            **client_kwargs,
        )

        if async_client_kwargs is None:
            async_client_kwargs = {}
        self.async_client = AsyncAnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
            **async_client_kwargs,
        )

    def count_tokens(self, content: str | Dict) -> int:
        """
        Override count_tokens since AnthropicBedrock doesn't have this method.
        Use tiktoken as a fallback for token estimation.
        """
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use GPT-3.5 as approximation
        
        if isinstance(content, str):
            return len(enc.encode(content, disallowed_special=()))
        
        # Handle message format
        formatting_token_count = 4
        total = 0
        for message in content:
            total += len(enc.encode(message["content"], disallowed_special=())) + formatting_token_count
        return total

    @property
    def support_message_api(self):
        return "claude-3" in self.model