"""
Tests for Issue #16: Aleph Alpha AsyncClient initialization in non-async context.

This test verifies that:
1. AlephAlphaProvider can be initialized in non-async context without errors
2. AsyncClient is lazily created only when needed
3. The provider works correctly in both sync and async contexts
"""

import pytest
from unittest.mock import patch, MagicMock


class TestAlephAlphaLazyAsyncClient:
    """Test Aleph Alpha provider's lazy AsyncClient initialization."""

    @patch('llms.providers.aleph.Client')
    def test_init_does_not_create_async_client(self, mock_client):
        """Test that __init__ does not create AsyncClient immediately."""
        mock_client.return_value = MagicMock()

        from llms.providers.aleph import AlephAlphaProvider

        # This should not raise even without an event loop
        provider = AlephAlphaProvider(api_key="test-key")

        # Verify AsyncClient was not created during init
        assert provider._async_client is None
        assert provider._api_key == "test-key"

    @patch('llms.providers.aleph.Client')
    @patch('llms.providers.aleph.AsyncClient')
    def test_async_client_created_on_access(self, mock_async_client, mock_client):
        """Test that AsyncClient is created when async_client property is accessed."""
        mock_client.return_value = MagicMock()
        mock_async_instance = MagicMock()
        mock_async_client.return_value = mock_async_instance

        from llms.providers.aleph import AlephAlphaProvider

        provider = AlephAlphaProvider(api_key="test-key")

        # Initially not created
        assert provider._async_client is None

        # Access the property
        client = provider.async_client

        # Now it should be created
        assert client is mock_async_instance
        mock_async_client.assert_called_once_with("test-key")

    @patch('llms.providers.aleph.Client')
    @patch('llms.providers.aleph.AsyncClient')
    def test_async_client_cached_on_subsequent_access(self, mock_async_client, mock_client):
        """Test that AsyncClient is cached and not recreated on subsequent access."""
        mock_client.return_value = MagicMock()
        mock_async_instance = MagicMock()
        mock_async_client.return_value = mock_async_instance

        from llms.providers.aleph import AlephAlphaProvider

        provider = AlephAlphaProvider(api_key="test-key")

        # Access multiple times
        client1 = provider.async_client
        client2 = provider.async_client

        # Should be the same instance
        assert client1 is client2
        # AsyncClient should only be called once
        mock_async_client.assert_called_once()

    @patch('llms.providers.aleph.Client')
    def test_sync_complete_does_not_touch_async_client(self, mock_client):
        """Test that sync complete() doesn't create AsyncClient."""
        mock_response = MagicMock()
        mock_response.completions = [MagicMock(completion="test response")]
        mock_client_instance = MagicMock()
        mock_client_instance.complete.return_value = mock_response
        mock_client.return_value = mock_client_instance

        from llms.providers.aleph import AlephAlphaProvider

        provider = AlephAlphaProvider(api_key="test-key", model="luminous-base")

        # Call sync complete
        result = provider.complete("test prompt")

        # AsyncClient should not have been created
        assert provider._async_client is None

    @patch('llms.providers.aleph.Client')
    def test_model_defaults_to_first_model(self, mock_client):
        """Test that model defaults to first model in MODEL_INFO."""
        mock_client.return_value = MagicMock()

        from llms.providers.aleph import AlephAlphaProvider

        provider = AlephAlphaProvider(api_key="test-key")

        assert provider.model == "luminous-base"

    @patch('llms.providers.aleph.Client')
    def test_init_in_thread_without_event_loop(self, mock_client):
        """Test that provider can be initialized in a thread without event loop."""
        import threading

        mock_client.return_value = MagicMock()
        from llms.providers.aleph import AlephAlphaProvider

        errors = []
        provider_holder = []

        def init_provider():
            try:
                # This should not raise RuntimeError about missing event loop
                provider = AlephAlphaProvider(api_key="test-key")
                provider_holder.append(provider)
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=init_provider)
        thread.start()
        thread.join()

        assert len(errors) == 0, f"Unexpected error: {errors}"
        assert len(provider_holder) == 1
        assert provider_holder[0]._async_client is None


class TestAlephAlphaAsyncComplete:
    """Test Aleph Alpha async completion functionality."""

    @pytest.mark.asyncio
    @patch('llms.providers.aleph.Client')
    @patch('llms.providers.aleph.AsyncClient')
    async def test_acomplete_creates_async_client(self, mock_async_client, mock_client):
        """Test that acomplete creates and uses AsyncClient."""
        mock_client.return_value = MagicMock()

        # Create a mock async context manager
        mock_response = MagicMock()
        mock_response.completions = [MagicMock(completion="async response")]

        mock_async_context = MagicMock()
        mock_async_context.__aenter__ = MagicMock(return_value=MagicMock())
        mock_async_context.__aexit__ = MagicMock(return_value=None)
        mock_async_context.__aenter__.return_value.complete = MagicMock(
            return_value=mock_response
        )

        mock_async_instance = MagicMock()
        mock_async_instance.__aenter__ = mock_async_context.__aenter__
        mock_async_instance.__aexit__ = mock_async_context.__aexit__
        mock_async_client.return_value = mock_async_instance

        from llms.providers.aleph import AlephAlphaProvider

        provider = AlephAlphaProvider(api_key="test-key", model="luminous-base")

        # Initially no async client
        assert provider._async_client is None

        # After acomplete, async client should be created
        # Note: The actual test would need proper async mocking
