from __future__ import annotations

from dataclasses import dataclass

from .providers import (
    AI21Provider,
    AlephAlphaProvider,
    AnthropicProvider,
    BedrockAnthropicProvider,
    CohereProvider,
    DeepSeekProvider,
    GoogleGenAIProvider,
    GoogleProvider,
    GroqProvider,
    HuggingfaceHubProvider,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    RekaProvider,
    TogetherProvider,
)
from .providers.base_provider import BaseProvider


@dataclass
class Provider:
    provider: type[BaseProvider]
    api_key_name: str | None = None
    api_key: str | None = None
    needs_api_key: bool = True


def create_provider(
    provider_class: type[BaseProvider],
    api_key_name: str | None = None,
    needs_api_key: bool = True,
) -> Provider:
    return Provider(provider_class, api_key_name=api_key_name, needs_api_key=needs_api_key)


PROVIDER_MAP = {
    "OpenAI": create_provider(OpenAIProvider, "OPENAI_API_KEY"),
    "Anthropic": create_provider(AnthropicProvider, "ANTHROPIC_API_KEY"),
    "BedrockAnthropic": create_provider(BedrockAnthropicProvider, needs_api_key=False),
    "AI21": create_provider(AI21Provider, "AI21_API_KEY"),
    "Cohere": create_provider(CohereProvider, "COHERE_API_KEY"),
    "AlephAlpha": create_provider(AlephAlphaProvider, "ALEPHALPHA_API_KEY"),
    "HuggingfaceHub": create_provider(HuggingfaceHubProvider, "HUGGINFACEHUB_API_KEY"),
    "GoogleGenAI": create_provider(GoogleGenAIProvider, "GOOGLE_API_KEY"),
    "Mistral": create_provider(MistralProvider, "MISTRAL_API_KEY"),
    "Google": create_provider(GoogleProvider, needs_api_key=False),
    "Ollama": create_provider(OllamaProvider, needs_api_key=False),
    "DeepSeek": create_provider(DeepSeekProvider, "DEEPSEEK_API_KEY"),
    "Groq": create_provider(GroqProvider, "GROQ_API_KEY"),
    "Reka": create_provider(RekaProvider, "REKA_API_KEY"),
    "Together": create_provider(TogetherProvider, "TOGETHER_API_KEY"),
    "OpenRouter": create_provider(OpenRouterProvider, "OPENROUTER_API_KEY"),
}
