"""
Tests for Issue #30: Ollama provider with custom host support.

This test verifies that:
1. OllamaProvider.MODEL_INFO can be updated with custom host
2. The model validation works correctly with custom ollama_host
3. The class-level cache works as expected
"""

import pytest
from unittest.mock import patch, MagicMock

from llms.providers.ollama import OllamaProvider, _get_model_info


class TestOllamaCustomHost:
    """Test Ollama provider with custom host configuration."""

    def setup_method(self):
        """Reset the MODEL_INFO cache before each test."""
        OllamaProvider._model_info_cache = {}
        # Store original MODEL_INFO to restore later
        self._original_model_info = OllamaProvider.MODEL_INFO.copy()

    def teardown_method(self):
        """Restore original MODEL_INFO after each test."""
        OllamaProvider.MODEL_INFO = self._original_model_info
        OllamaProvider._model_info_cache = {}

    def test_get_model_info_with_default_host(self):
        """Test _get_model_info function with default host."""
        # This should not raise an error even if Ollama is not running
        result = _get_model_info()
        assert isinstance(result, dict)

    def test_get_model_info_with_custom_host(self):
        """Test _get_model_info function with custom host."""
        custom_host = "http://custom-ollama-server:11434"
        result = _get_model_info(custom_host)
        assert isinstance(result, dict)

    @patch('llms.providers.ollama.Client')
    def test_get_model_info_for_host_caching(self, mock_client):
        """Test that model info is cached per host."""
        mock_instance = MagicMock()
        mock_instance.list.return_value = {
            "models": [{"name": "test-model:latest"}]
        }
        mock_client.return_value = mock_instance

        host = "http://test-host:11434"

        # First call should populate cache
        result1 = OllamaProvider.get_model_info_for_host(host)
        assert "test-model:latest" in result1

        # Second call should use cache
        result2 = OllamaProvider.get_model_info_for_host(host)
        assert result1 is result2

        # Client should only be called once due to caching
        assert mock_client.call_count == 1

    @patch('llms.providers.ollama.Client')
    def test_update_model_info_updates_class_level(self, mock_client):
        """Test that update_model_info updates the class-level MODEL_INFO."""
        mock_instance = MagicMock()
        mock_instance.list.return_value = {
            "models": [{"name": "custom-model:latest"}]
        }
        mock_client.return_value = mock_instance

        host = "http://custom-host:11434"
        OllamaProvider.update_model_info(host)

        assert "custom-model:latest" in OllamaProvider.MODEL_INFO

    @patch('llms.providers.ollama.Client')
    @patch('llms.providers.ollama.AsyncClient')
    def test_init_with_custom_host_updates_model_info(self, mock_async_client, mock_client):
        """Test that __init__ with custom host updates MODEL_INFO."""
        mock_instance = MagicMock()
        mock_instance.list.return_value = {
            "models": [{"name": "remote-model:latest"}]
        }
        mock_client.return_value = mock_instance

        custom_host = "http://remote-server:11434"
        provider = OllamaProvider(model="remote-model:latest", ollama_host=custom_host)

        assert provider.model == "remote-model:latest"
        assert provider.ollama_host == custom_host
        assert "remote-model:latest" in OllamaProvider.MODEL_INFO

    @patch('llms.providers.ollama.Client')
    @patch('llms.providers.ollama.AsyncClient')
    def test_init_without_model_uses_first_available(self, mock_async_client, mock_client):
        """Test that __init__ without model parameter uses first available model."""
        mock_instance = MagicMock()
        mock_instance.list.return_value = {
            "models": [
                {"name": "first-model:latest"},
                {"name": "second-model:latest"}
            ]
        }
        mock_client.return_value = mock_instance

        provider = OllamaProvider(ollama_host="http://test:11434")

        assert provider.model == "first-model:latest"

    @patch('llms.providers.ollama.Client')
    def test_init_raises_error_when_no_models(self, mock_client):
        """Test that __init__ raises ValueError when no models are available."""
        mock_instance = MagicMock()
        mock_instance.list.return_value = {"models": []}
        mock_client.return_value = mock_instance

        # Clear the cache to force fetching from the mock
        OllamaProvider._model_info_cache = {}

        with pytest.raises(ValueError, match="No models found"):
            OllamaProvider(ollama_host="http://empty-server:11434")

    @patch('llms.providers.ollama.Client')
    @patch('llms.providers.ollama.AsyncClient')
    def test_different_hosts_have_separate_models(self, mock_async_client, mock_client):
        """Test that different hosts maintain separate model lists."""
        call_count = [0]

        def create_mock_client(host=None, **kwargs):
            mock = MagicMock()
            call_count[0] += 1
            if "host1" in host:
                mock.list.return_value = {"models": [{"name": "model-a:latest"}]}
            else:
                mock.list.return_value = {"models": [{"name": "model-b:latest"}]}
            return mock

        mock_client.side_effect = create_mock_client

        # Clear cache
        OllamaProvider._model_info_cache = {}

        host1 = "http://host1:11434"
        host2 = "http://host2:11434"

        info1 = OllamaProvider.get_model_info_for_host(host1)
        info2 = OllamaProvider.get_model_info_for_host(host2)

        assert "model-a:latest" in info1
        assert "model-a:latest" not in info2
        assert "model-b:latest" in info2
        assert "model-b:latest" not in info1


class TestOllamaLLMSIntegration:
    """Test Ollama integration with LLMS class."""

    def setup_method(self):
        """Reset the MODEL_INFO cache before each test."""
        OllamaProvider._model_info_cache = {}
        self._original_model_info = OllamaProvider.MODEL_INFO.copy()

    def teardown_method(self):
        """Restore original MODEL_INFO after each test."""
        OllamaProvider.MODEL_INFO = self._original_model_info
        OllamaProvider._model_info_cache = {}

    @patch('llms.providers.ollama.Client')
    @patch('llms.providers.ollama.AsyncClient')
    def test_llms_init_with_ollama_custom_host(self, mock_async_client, mock_client):
        """Test that llms.init works with ollama_host parameter."""
        mock_instance = MagicMock()
        mock_instance.list.return_value = {
            "models": [{"name": "llama2:latest"}]
        }
        mock_client.return_value = mock_instance

        import llms

        try:
            model = llms.init(
                model="llama2:latest",
                ollama_host="http://custom-ollama:11434"
            )
            assert model is not None
            assert len(model._providers) == 1
            assert model._providers[0].model == "llama2:latest"
        except ValueError:
            # If no other providers match, this is expected
            pass

    @patch('llms.providers.ollama.Client')
    def test_validate_model_updates_model_info(self, mock_client):
        """Test that _validate_model updates MODEL_INFO for Ollama."""
        mock_instance = MagicMock()
        mock_instance.list.return_value = {
            "models": [{"name": "custom-llama:latest"}]
        }
        mock_client.return_value = mock_instance

        from llms.llms import LLMS, Provider

        # Clear cache
        OllamaProvider._model_info_cache = {}

        provider = Provider(
            provider=OllamaProvider,
            needs_api_key=False
        )

        kwargs = {"ollama_host": "http://custom-server:11434"}

        # Create a minimal LLMS instance for testing
        class TestLLMS(LLMS):
            def _initialize_providers(self, kwargs):
                self._providers = []

        test_llms = TestLLMS()

        # Call _validate_model with custom ollama_host
        result = test_llms._validate_model("custom-llama:latest", provider, kwargs)

        assert "custom-llama:latest" in OllamaProvider.MODEL_INFO
        assert result is True
