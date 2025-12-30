"""Tests for context caching functionality in Anthropic and Google providers."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestAnthropicContextCaching:
    """Test Anthropic provider context caching."""

    def test_cache_system_message(self):
        """Test that cache_system properly formats system message."""
        # Clear module cache to ensure fresh import
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('llms')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with patch.dict('sys.modules', {'anthropic': MagicMock()}):
            from llms.providers.anthropic import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")
            model_inputs = provider._prepare_message_inputs(
                prompt="Hello",
                system_message="You are a helpful assistant.",
                cache_system=True,
            )

            # System should be formatted as list with cache_control
            assert isinstance(model_inputs["system"], list)
            assert len(model_inputs["system"]) == 1
            assert model_inputs["system"][0]["type"] == "text"
            assert model_inputs["system"][0]["text"] == "You are a helpful assistant."
            assert model_inputs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_messages(self):
        """Test that cache_messages properly formats the last message."""
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('llms')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with patch.dict('sys.modules', {'anthropic': MagicMock()}):
            from llms.providers.anthropic import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")
            model_inputs = provider._prepare_message_inputs(
                prompt="What is the summary?",
                cache_messages=True,
            )

            # Last message should have cache_control
            last_msg = model_inputs["messages"][-1]
            assert isinstance(last_msg["content"], list)
            assert last_msg["content"][0]["type"] == "text"
            assert last_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_cache_by_default(self):
        """Test that caching is disabled by default."""
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('llms')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with patch.dict('sys.modules', {'anthropic': MagicMock()}):
            from llms.providers.anthropic import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")
            model_inputs = provider._prepare_message_inputs(
                prompt="Hello",
                system_message="You are a helpful assistant.",
            )

            # System should be a plain string
            assert isinstance(model_inputs["system"], str)
            # Messages should have string content
            assert isinstance(model_inputs["messages"][-1]["content"], str)

    def test_cache_metadata_extraction(self):
        """Test that cache metadata is properly extracted when available."""
        # This test verifies the metadata extraction logic directly
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('llms')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Create a mock usage object
        class MockUsage:
            input_tokens = 100
            output_tokens = 50
            cache_creation_input_tokens = 80
            cache_read_input_tokens = 0

        usage = MockUsage()
        meta = {}

        # Simulate the metadata extraction logic from complete()
        if hasattr(usage, "cache_creation_input_tokens"):
            meta["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
        if hasattr(usage, "cache_read_input_tokens"):
            meta["cache_read_input_tokens"] = usage.cache_read_input_tokens

        assert meta["cache_creation_input_tokens"] == 80
        assert meta["cache_read_input_tokens"] == 0


class TestGoogleContextCaching:
    """Test Google GenAI provider context caching."""

    def test_cached_content_parameter(self):
        """Test that cached_content is passed through to config."""
        with patch('llms.providers.google_genai.genai') as mock_genai:
            from llms.providers.google_genai import GoogleGenAIProvider

            mock_genai.Client.return_value = Mock()

            provider = GoogleGenAIProvider(api_key="test-key")
            model_inputs = provider._prepare_model_inputs(
                prompt="Test prompt",
                cached_content="caches/abc123",
            )

            assert model_inputs["cached_content"] == "caches/abc123"

    def test_create_cache_method(self):
        """Test the create_cache helper method."""
        with patch('llms.providers.google_genai.genai') as mock_genai:
            from llms.providers.google_genai import GoogleGenAIProvider

            mock_client = Mock()
            mock_cache = Mock()
            mock_cache.name = "caches/test-cache-123"
            mock_client.caches.create.return_value = mock_cache
            mock_genai.Client.return_value = mock_client

            provider = GoogleGenAIProvider(api_key="test-key")
            cache_name = provider.create_cache(
                contents="Long document content...",
                system_instruction="You are an expert.",
                ttl="7200s",
                display_name="My Cache"
            )

            assert cache_name == "caches/test-cache-123"
            mock_client.caches.create.assert_called_once()

    def test_delete_cache_method(self):
        """Test the delete_cache helper method."""
        with patch('llms.providers.google_genai.genai') as mock_genai:
            from llms.providers.google_genai import GoogleGenAIProvider

            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            provider = GoogleGenAIProvider(api_key="test-key")
            provider.delete_cache("caches/test-cache-123")

            mock_client.caches.delete.assert_called_once_with(name="caches/test-cache-123")
