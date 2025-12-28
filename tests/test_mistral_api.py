"""
Tests for Issue #23: Mistral provider compatibility with new API.

This test verifies that:
1. MistralProvider can be imported without errors
2. The new mistralai SDK (>=1.0.0) is compatible with the provider
3. Basic initialization works correctly
"""

import pytest
from unittest.mock import patch, MagicMock


class TestMistralProviderImport:
    """Test that Mistral provider imports correctly with new API."""

    def test_mistralai_import(self):
        """Test that mistralai can be imported with new API."""
        from mistralai import Mistral
        assert Mistral is not None

    def test_mistral_provider_import(self):
        """Test that MistralProvider can be imported."""
        from llms.providers.mistral import MistralProvider
        assert MistralProvider is not None

    def test_mistral_provider_has_model_info(self):
        """Test that MistralProvider has MODEL_INFO defined."""
        from llms.providers.mistral import MistralProvider
        assert hasattr(MistralProvider, 'MODEL_INFO')
        assert len(MistralProvider.MODEL_INFO) > 0


class TestMistralProviderInitialization:
    """Test Mistral provider initialization."""

    @patch('llms.providers.mistral.Mistral')
    def test_init_with_api_key(self, mock_mistral):
        """Test that provider can be initialized with API key."""
        mock_mistral.return_value = MagicMock()

        from llms.providers.mistral import MistralProvider

        provider = MistralProvider(api_key="test-key")

        assert provider is not None
        assert provider.model is not None
        mock_mistral.assert_called_once_with(api_key="test-key")

    @patch('llms.providers.mistral.Mistral')
    def test_init_with_specific_model(self, mock_mistral):
        """Test that provider can be initialized with specific model."""
        mock_mistral.return_value = MagicMock()

        from llms.providers.mistral import MistralProvider

        provider = MistralProvider(
            api_key="test-key",
            model="mistral-large-latest"
        )

        assert provider.model == "mistral-large-latest"

    @patch('llms.providers.mistral.Mistral')
    def test_init_with_client_kwargs(self, mock_mistral):
        """Test that provider can be initialized with custom client kwargs."""
        mock_mistral.return_value = MagicMock()

        from llms.providers.mistral import MistralProvider

        provider = MistralProvider(
            api_key="test-key",
            client_kwargs={"timeout": 30}
        )

        mock_mistral.assert_called_once_with(api_key="test-key", timeout=30)


class TestMistralProviderMethods:
    """Test Mistral provider methods."""

    @patch('llms.providers.mistral.Mistral')
    def test_count_tokens(self, mock_mistral):
        """Test token counting functionality."""
        mock_mistral.return_value = MagicMock()

        from llms.providers.mistral import MistralProvider

        provider = MistralProvider(api_key="test-key")
        token_count = provider.count_tokens("Hello, world!")

        assert isinstance(token_count, int)
        assert token_count > 0

    @patch('llms.providers.mistral.Mistral')
    def test_count_tokens_with_messages(self, mock_mistral):
        """Test token counting with message list."""
        mock_mistral.return_value = MagicMock()

        from llms.providers.mistral import MistralProvider

        provider = MistralProvider(api_key="test-key")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        token_count = provider.count_tokens(messages)

        assert isinstance(token_count, int)
        assert token_count > 0

    @patch('llms.providers.mistral.Mistral')
    def test_prepare_model_inputs(self, mock_mistral):
        """Test model input preparation."""
        mock_mistral.return_value = MagicMock()

        from llms.providers.mistral import MistralProvider

        provider = MistralProvider(api_key="test-key")
        inputs = provider._prepare_model_inputs(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100
        )

        assert "messages" in inputs
        assert inputs["temperature"] == 0.5
        assert inputs["max_tokens"] == 100
        assert len(inputs["messages"]) == 1
        assert inputs["messages"][0]["role"] == "user"
        assert inputs["messages"][0]["content"] == "Test prompt"

    @patch('llms.providers.mistral.Mistral')
    def test_prepare_model_inputs_with_system_message(self, mock_mistral):
        """Test model input preparation with system message."""
        mock_mistral.return_value = MagicMock()

        from llms.providers.mistral import MistralProvider

        provider = MistralProvider(api_key="test-key")
        inputs = provider._prepare_model_inputs(
            prompt="Test prompt",
            system_message="You are a helpful assistant."
        )

        assert len(inputs["messages"]) == 2
        assert inputs["messages"][0]["role"] == "system"
        assert inputs["messages"][0]["content"] == "You are a helpful assistant."

    @patch('llms.providers.mistral.Mistral')
    def test_prepare_model_inputs_with_history(self, mock_mistral):
        """Test model input preparation with conversation history."""
        mock_mistral.return_value = MagicMock()

        from llms.providers.mistral import MistralProvider

        provider = MistralProvider(api_key="test-key")
        history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ]
        inputs = provider._prepare_model_inputs(
            prompt="New prompt",
            history=history
        )

        assert len(inputs["messages"]) == 3
        assert inputs["messages"][0]["content"] == "Previous message"
        assert inputs["messages"][1]["content"] == "Previous response"
        assert inputs["messages"][2]["content"] == "New prompt"


class TestMistralModelInfo:
    """Test Mistral model information."""

    def test_all_models_have_required_fields(self):
        """Test that all models have required pricing and limit fields."""
        from llms.providers.mistral import MistralProvider

        for model_name, model_info in MistralProvider.MODEL_INFO.items():
            assert "prompt" in model_info, f"{model_name} missing 'prompt' field"
            assert "completion" in model_info, f"{model_name} missing 'completion' field"
            assert "token_limit" in model_info, f"{model_name} missing 'token_limit' field"
            assert isinstance(model_info["prompt"], (int, float))
            assert isinstance(model_info["completion"], (int, float))
            assert isinstance(model_info["token_limit"], int)
