"""
Tests for custom api_base functionality.
"""

import pytest


class TestOpenAICustomApiBase:
    """Test OpenAI provider with custom api_base."""

    def test_openai_provider_accepts_api_base(self):
        """Test that OpenAIProvider accepts api_base parameter."""
        from llms.providers.openai import OpenAIProvider

        # Should not raise an error
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o",
            api_base="https://custom.openai.com/v1"
        )
        assert provider.model == "gpt-4o"
        assert provider.client.base_url.host == "custom.openai.com"

    def test_openai_provider_default_base_url(self):
        """Test that OpenAIProvider uses default base URL when api_base not set."""
        from llms.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert "openai.com" in str(provider.client.base_url)


class TestDeepSeekCustomApiBase:
    """Test DeepSeek provider with custom api_base."""

    def test_deepseek_provider_accepts_api_base(self):
        """Test that DeepSeekProvider accepts api_base parameter."""
        from llms.providers.deepseek import DeepSeekProvider

        provider = DeepSeekProvider(
            api_key="test-key",
            model="deepseek-chat",
            api_base="https://custom.deepseek.com/v1"
        )
        assert provider.model == "deepseek-chat"
        assert provider.client.base_url.host == "custom.deepseek.com"

    def test_deepseek_provider_default_base_url(self):
        """Test that DeepSeekProvider uses default base URL when api_base not set."""
        from llms.providers.deepseek import DeepSeekProvider

        provider = DeepSeekProvider(api_key="test-key", model="deepseek-chat")
        assert "api.deepseek.com" in str(provider.client.base_url)


class TestGroqCustomApiBase:
    """Test Groq provider with custom api_base."""

    def test_groq_provider_accepts_api_base(self):
        """Test that GroqProvider accepts api_base parameter."""
        from llms.providers.groq import GroqProvider

        provider = GroqProvider(
            api_key="test-key",
            model="llama-3.1-8b-instant",
            api_base="https://custom.groq.com/v1"
        )
        assert provider.model == "llama-3.1-8b-instant"
        assert provider.client.base_url.host == "custom.groq.com"

    def test_groq_provider_default_base_url(self):
        """Test that GroqProvider uses default base URL when api_base not set."""
        from llms.providers.groq import GroqProvider

        provider = GroqProvider(api_key="test-key", model="llama-3.1-8b-instant")
        assert "api.groq.com" in str(provider.client.base_url)


class TestOpenRouterCustomApiBase:
    """Test OpenRouter provider with custom api_base."""

    def test_openrouter_provider_accepts_api_base(self):
        """Test that OpenRouterProvider accepts api_base parameter."""
        from llms.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(
            api_key="test-key",
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            api_base="https://custom.openrouter.ai/v1"
        )
        assert provider.model == "nvidia/llama-3.1-nemotron-70b-instruct"
        assert provider.client.base_url.host == "custom.openrouter.ai"

    def test_openrouter_provider_default_base_url(self):
        """Test that OpenRouterProvider uses default base URL when api_base not set."""
        from llms.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(
            api_key="test-key",
            model="nvidia/llama-3.1-nemotron-70b-instruct"
        )
        assert "openrouter.ai" in str(provider.client.base_url)
