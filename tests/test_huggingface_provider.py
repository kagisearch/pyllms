"""Tests for HuggingFace Hub Provider using InferenceClient."""

import pytest
from unittest.mock import Mock, patch

from llms.providers.huggingface import HuggingfaceHubProvider


class TestHuggingfaceProviderInit:
    """Test HuggingfaceHubProvider initialization."""

    def test_default_model(self):
        """Test that default model is set correctly."""
        with patch('llms.providers.huggingface.InferenceClient'):
            provider = HuggingfaceHubProvider(api_key="test-key")
            assert provider.model == "hf_pythia"
            assert provider._model_full_name == "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"

    def test_custom_model(self):
        """Test initialization with custom model."""
        with patch('llms.providers.huggingface.InferenceClient'):
            provider = HuggingfaceHubProvider(api_key="test-key", model="hf_falcon7b")
            assert provider.model == "hf_falcon7b"
            assert provider._model_full_name == "tiiuae/falcon-7b-instruct"

    def test_inference_client_created(self):
        """Test that InferenceClient is created with token."""
        with patch('llms.providers.huggingface.InferenceClient') as mock_client:
            provider = HuggingfaceHubProvider(api_key="test-api-key")
            mock_client.assert_called_once_with(token="test-api-key")


class TestHuggingfaceProviderComplete:
    """Test HuggingfaceHubProvider complete method."""

    def test_complete_calls_text_generation(self):
        """Test that complete uses text_generation API."""
        with patch('llms.providers.huggingface.InferenceClient') as mock_client_class:
            mock_client = Mock()
            # For hf_pythia, the prompt is formatted with special tokens
            formatted_prompt = "<|prompter|Test prompt.<|endoftext|><|assistant|>"
            mock_client.text_generation.return_value = formatted_prompt + " This is the response."
            mock_client_class.return_value = mock_client

            provider = HuggingfaceHubProvider(api_key="test-key")
            result = provider.complete("Test prompt.", max_tokens=100)

            mock_client.text_generation.assert_called_once()
            call_kwargs = mock_client.text_generation.call_args
            # hf_pythia model wraps prompt with special tokens
            assert "<|prompter|" in call_kwargs[0][0]
            assert call_kwargs[1]["model"] == "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
            assert call_kwargs[1]["return_full_text"] is True

    def test_complete_with_falcon_model(self):
        """Test complete with a model that doesn't modify prompt."""
        with patch('llms.providers.huggingface.InferenceClient') as mock_client_class:
            mock_client = Mock()
            mock_client.text_generation.return_value = "Hello world. This is a response."
            mock_client_class.return_value = mock_client

            provider = HuggingfaceHubProvider(api_key="test-key", model="hf_falcon7b")
            result = provider.complete("Hello world.", max_tokens=100)

            call_kwargs = mock_client.text_generation.call_args
            assert call_kwargs[0][0] == "Hello world."
            assert call_kwargs[1]["model"] == "tiiuae/falcon-7b-instruct"

    def test_complete_returns_result(self):
        """Test that complete returns a valid Result object."""
        with patch('llms.providers.huggingface.InferenceClient') as mock_client_class:
            mock_client = Mock()
            mock_client.text_generation.return_value = "Hello world. This is a response."
            mock_client_class.return_value = mock_client

            provider = HuggingfaceHubProvider(api_key="test-key", model="hf_falcon7b")
            result = provider.complete("Hello world.")

            assert result.text == " This is a response."
            assert result.meta["tokens_prompt"] == -1
            assert result.meta["tokens_completion"] == -1
            assert "latency" in result.meta


class TestHuggingfaceModelInfo:
    """Test MODEL_INFO structure."""

    def test_all_models_have_required_fields(self):
        """Test that all models have required pricing fields."""
        for model_name, model_info in HuggingfaceHubProvider.MODEL_INFO.items():
            assert "full" in model_info, f"{model_name} missing 'full' name"
            assert "prompt" in model_info, f"{model_name} missing 'prompt' price"
            assert "completion" in model_info, f"{model_name} missing 'completion' price"
            assert "token_limit" in model_info, f"{model_name} missing 'token_limit'"
