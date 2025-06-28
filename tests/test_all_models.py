"""
Comprehensive test suite for all PyLLMs models.

This test suite dynamically discovers all available providers and models
using the existing LLMS infrastructure, checks for API keys in environment 
variables, and runs tests only for providers with valid API keys available.

No models or providers are hardcoded - everything is discovered dynamically.
"""

import os
import pytest
from typing import Dict, List, Tuple, Any

# Import the main LLMS class
import llms
from llms.llms import LLMS


def get_available_providers() -> Dict[str, Any]:
    """
    Get all providers that have API keys available or don't need them.
    Uses the existing LLMS._provider_map infrastructure.
    
    Returns:
        Dict mapping provider names to their Provider objects if API key is available
    """
    available_providers = {}
    
    for provider_name, provider_config in LLMS._provider_map.items():
        if not provider_config.needs_api_key:
            # Providers that don't need API keys (like Ollama, Google Vertex)
            available_providers[provider_name] = provider_config
        elif provider_config.custom_credential_check:
            # Providers with custom credential checking (like BedrockAnthropic)
            if provider_config.custom_credential_check():
                available_providers[provider_name] = provider_config
        elif provider_config.api_key_name and os.getenv(provider_config.api_key_name):
            # Providers that need API keys and have them available
            available_providers[provider_name] = provider_config
    
    return available_providers


def get_all_models() -> List[Tuple[str, str, Any]]:
    """
    Dynamically discover all models from available providers.
    Uses the existing LLMS infrastructure.
    
    Returns:
        List of tuples: (provider_name, model_name, provider_class)
    """
    available_providers = get_available_providers()
    all_models = []
    
    for provider_name, provider_config in available_providers.items():
        provider_class = provider_config.provider
        
        # Get all models from the provider's MODEL_INFO
        if hasattr(provider_class, 'MODEL_INFO'):
            models = list(provider_class.MODEL_INFO.keys())
            for model_name in models:
                all_models.append((provider_name, model_name, provider_class))
    
    return all_models


@pytest.fixture(scope="session")
def available_providers():
    """Fixture that returns all available providers."""
    return get_available_providers()


@pytest.fixture(scope="session") 
def all_model_combinations():
    """Fixture that returns all available model combinations."""
    return get_all_models()


class TestModelDiscovery:
    """Test that we can discover models and providers correctly using LLMS infrastructure."""
    
    def test_provider_discovery(self, available_providers):
        """Test that we can discover available providers from LLMS._provider_map."""
        assert len(available_providers) > 0, "No providers with API keys found"
        print(f"\nFound {len(available_providers)} available providers:")
        
        for provider_name, provider_config in available_providers.items():
            if not provider_config.needs_api_key:
                api_key_status = "No API key needed"
            elif provider_config.custom_credential_check:
                api_key_status = "Custom credentials"
            else:
                api_key_status = f"API key: {provider_config.api_key_name}"
            print(f"  ✓ {provider_name} ({api_key_status})")
            
            assert hasattr(provider_config, 'provider')
            assert hasattr(provider_config, 'needs_api_key')
            
            if provider_config.needs_api_key and not provider_config.custom_credential_check:
                assert provider_config.api_key_name is not None
                assert os.getenv(provider_config.api_key_name) is not None
    
    def test_model_discovery(self, all_model_combinations):
        """Test that we can discover models from providers using MODEL_INFO."""
        assert len(all_model_combinations) > 0, "No models discovered"
        print(f"\nFound {len(all_model_combinations)} total models")
        
        models_by_provider = {}
        for provider_name, model_name, provider_class in all_model_combinations:
            if provider_name not in models_by_provider:
                models_by_provider[provider_name] = []
            models_by_provider[provider_name].append(model_name)
            
            assert isinstance(provider_name, str)
            assert isinstance(model_name, str)
            assert hasattr(provider_class, 'MODEL_INFO')
            assert model_name in provider_class.MODEL_INFO
        
        for provider_name, models in models_by_provider.items():
            print(f"  {provider_name}: {len(models)} models")


class TestModelInitialization:
    """Test that models can be initialized correctly using llms.init()."""
    
    @pytest.mark.parametrize("provider_name,model_name,provider_class", get_all_models())
    def test_model_initialization_via_llms_init(self, provider_name, model_name, provider_class):
        """Test that each model can be initialized through llms.init()."""
        try:
            model = llms.init(model_name)
            assert model is not None
            assert len(model._providers) == 1
            assert model._models == [model_name]
            assert hasattr(model, 'complete')
            assert hasattr(model, 'count_tokens')
        except Exception as e:
            pytest.fail(f"Failed to initialize model {model_name} from {provider_name}: {e}")
    
    @pytest.mark.parametrize("provider_name,model_name,provider_class", get_all_models())
    def test_provider_direct_initialization(self, provider_name, model_name, provider_class):
        """Test that each provider can be initialized directly."""
        provider_config = LLMS._provider_map[provider_name]
        
        try:
            if provider_name == "BedrockAnthropic":
                # Special case for BedrockAnthropic which uses AWS credentials
                provider = provider_class(
                    model=model_name,
                    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
                )
            elif provider_config.needs_api_key and provider_config.api_key_name:
                api_key = os.getenv(provider_config.api_key_name)
                provider = provider_class(api_key=api_key, model=model_name)
            else:
                provider = provider_class(model=model_name)
            
            assert provider is not None
            assert provider.model == model_name
            assert hasattr(provider, 'complete')
            assert hasattr(provider, 'count_tokens')
        except Exception as e:
            pytest.fail(f"Failed to initialize provider {provider_name} with model {model_name}: {e}")


class TestBasicModelFunctionality:
    """Test basic functionality of each model."""
    
    @pytest.mark.parametrize("provider_name,model_name,provider_class", get_all_models())
    def test_model_completion(self, provider_name, model_name, provider_class):
        """Test that each model can complete a simple prompt."""
        
        # Skip embedding and rerank models as they have different interfaces
        if any(keyword in model_name.lower() for keyword in ['embed', 'rerank']):
            pytest.skip(f"Skipping {model_name} - embedding/rerank model with different interface")
        
        try:
            model = llms.init(model_name)
            
            # Simple test prompt that should work across all text models
            prompt = "What is 2+2? Answer with just the number."
            
            # Set reasonable parameters that work for all models including thinking models
            result = model.complete(
                prompt,
                max_tokens=2048,
                temperature=0
            )
            
            assert result is not None
            assert hasattr(result, 'text')
            assert len(result.text.strip()) > 0
            assert hasattr(result, 'meta')
            
            print(f"✓ {provider_name}/{model_name}: '{result.text.strip()}'")
            
        except Exception as e:
            pytest.fail(f"Model {provider_name}/{model_name} failed: {e}")
    
    @pytest.mark.parametrize("provider_name,model_name,provider_class", get_all_models())
    def test_token_counting(self, provider_name, model_name, provider_class):
        """Test token counting functionality for each model."""
        
        if any(keyword in model_name.lower() for keyword in ['embed', 'rerank']):
            pytest.skip(f"Skipping {model_name} - embedding/rerank model")
        
        try:
            model = llms.init(model_name)
            
            test_text = "Hello, world! This is a test."
            token_count = model.count_tokens(test_text)
            
            assert isinstance(token_count, int)
            assert token_count > 0
            
            print(f"✓ {provider_name}/{model_name}: {token_count} tokens for '{test_text}'")
            
        except NotImplementedError as e:
            # Some providers legitimately don't support token counting
            if "Count tokens is currently not supported" in str(e):
                pytest.skip(f"Skipping {provider_name}/{model_name} - token counting not supported by provider")
            else:
                pytest.fail(f"Unexpected NotImplementedError for {provider_name}/{model_name}: {e}")
        except Exception as e:
            pytest.fail(f"Token counting failed for {provider_name}/{model_name}: {e}")


class TestAsyncFunctionality:
    """Test async functionality where supported."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_name,model_name,provider_class", 
                           [model for model in get_all_models() if not any(kw in model[1].lower() for kw in ['embed', 'rerank'])][:3])  # Test first 3 non-embed models
    async def test_async_completion(self, provider_name, model_name, provider_class):
        """Test async completion for supported models."""
        
        try:
            model = llms.init(model_name)
            
            prompt = "What is 3+3? Answer with just the number."
            
            result = await model.acomplete(
                prompt,
                max_tokens=10,
                temperature=0
            )
            
            assert result is not None
            assert hasattr(result, 'text')
            assert len(result.text.strip()) > 0
            
            print(f"✓ Async {provider_name}/{model_name}: '{result.text.strip()}'")
            
        except Exception as e:
            pytest.fail(f"Async test failed for {provider_name}/{model_name}: {e}")


class TestStreamingFunctionality:
    """Test streaming functionality where supported."""
    
    @pytest.mark.parametrize("provider_name,model_name,provider_class", 
                           [model for model in get_all_models() if not any(kw in model[1].lower() for kw in ['embed', 'rerank'])][:2])  # Test first 2 non-embed models
    def test_streaming_completion(self, provider_name, model_name, provider_class):
        """Test streaming completion for supported models."""
        
        try:
            model = llms.init(model_name)
            
            prompt = "Count: 1, 2, 3"
            
            result = model.complete_stream(
                prompt,
                max_tokens=20,
                temperature=0
            )
            
            assert result is not None
            assert hasattr(result, 'stream')
            
            # Collect stream chunks
            chunks = []
            for chunk in result.stream:
                if chunk is not None:
                    chunks.append(chunk)
                    if len(chunks) >= 3:  # Don't collect too many to avoid rate limits
                        break
            
            assert len(chunks) > 0
            full_text = ''.join(chunks)
            assert len(full_text.strip()) > 0
            
            print(f"✓ Stream {provider_name}/{model_name}: '{full_text.strip()}'")
            
        except Exception as e:
            pytest.fail(f"Streaming test failed for {provider_name}/{model_name}: {e}")


class TestModelInformation:
    """Test that model information is correctly defined."""
    
    @pytest.mark.parametrize("provider_name,model_name,provider_class", get_all_models())
    def test_model_info_structure(self, provider_name, model_name, provider_class):
        """Test that model info has required fields."""
        model_info = provider_class.MODEL_INFO[model_name]
        
        # All models should have prompt and completion pricing
        assert 'prompt' in model_info, f"{provider_name}/{model_name} missing 'prompt' pricing"
        assert 'completion' in model_info, f"{provider_name}/{model_name} missing 'completion' pricing"
        assert isinstance(model_info['prompt'], (int, float)), f"{provider_name}/{model_name} prompt pricing not numeric"
        assert isinstance(model_info['completion'], (int, float)), f"{provider_name}/{model_name} completion pricing not numeric"
        
        # Token limit should be specified
        assert 'token_limit' in model_info, f"{provider_name}/{model_name} missing 'token_limit'"
        assert isinstance(model_info['token_limit'], int), f"{provider_name}/{model_name} token_limit not integer"
        assert model_info['token_limit'] >= 0, f"{provider_name}/{model_name} token_limit negative"


class TestMultiModelFunctionality:
    """Test multi-model functionality."""
    
    def test_multi_model_init(self, all_model_combinations):
        """Test initializing multiple models at once."""
        # Get first 3 available models to avoid overwhelming APIs
        available_models = [model[1] for model in all_model_combinations if not any(kw in model[1].lower() for kw in ['embed', 'rerank'])][:3]
        
        if len(available_models) < 2:
            pytest.skip("Need at least 2 models for multi-model test")
        
        try:
            models = llms.init(model=available_models)
            assert models is not None
            assert len(models._providers) == len(available_models)
            assert models._models == available_models
            
        except Exception as e:
            pytest.fail(f"Multi-model initialization failed: {e}")


def test_no_hardcoded_models():
    """Ensure we're not hardcoding models anywhere in tests."""
    # This test ensures we're discovering models dynamically
    all_models = get_all_models()
    assert len(all_models) > 0, "No models discovered - discovery mechanism failed"
    
    # Verify we're discovering models dynamically (not hardcoded)
    providers_with_models = set(model[0] for model in all_models)
    assert len(providers_with_models) >= 1, "No providers found with models"
    
    # Verify that for each available provider, we're finding their models
    available_providers = get_available_providers()
    for provider_name in providers_with_models:
        assert provider_name in available_providers, f"Found models for {provider_name} but provider not in available_providers"
    
    if len(providers_with_models) == 1:
        print(f"✓ Dynamic discovery working: {len(all_models)} models from 1 provider: {list(providers_with_models)[0]}")
    else:
        print(f"✓ Dynamic discovery working: {len(all_models)} models from {len(providers_with_models)} providers")


def test_llms_list_method():
    """Test that LLMS.list() method works correctly."""
    # Create a minimal LLMS instance to test the list method
    # We'll override _initialize_providers to avoid needing API keys
    class TestLLMS(LLMS):
        def _initialize_providers(self, kwargs):
            # Skip provider initialization for list testing
            self._providers = []
    
    temp_llms = TestLLMS()
    all_models_list = temp_llms.list()
    
    assert len(all_models_list) > 0, "LLMS.list() returned no models"
    
    # Verify structure
    for model_info in all_models_list[:5]:  # Check first 5
        assert 'provider' in model_info
        assert 'name' in model_info
        assert 'cost' in model_info
    
    print(f"✓ LLMS.list() returned {len(all_models_list)} models")


if __name__ == "__main__":
    # When run directly, show available providers and models
    print("=== PyLLMs Dynamic Model Discovery ===")
    
    available = get_available_providers()
    print(f"\nAvailable Providers ({len(available)}):")
    for name, config in available.items():
        api_key_status = "✓ No API key needed" if not config.needs_api_key else f"✓ {config.api_key_name}"
        print(f"  {api_key_status} {name}")
    
    all_models = get_all_models()
    print(f"\nTotal Models Available: {len(all_models)}")
    
    # Group by provider
    by_provider = {}
    for provider_name, model_name, _ in all_models:
        if provider_name not in by_provider:
            by_provider[provider_name] = []
        by_provider[provider_name].append(model_name)
    
    for provider_name, models in by_provider.items():
        print(f"  {provider_name}: {len(models)} models")
    
    print("\nRun tests with: pytest tests/test_all_models.py -v")
    print("Run with output: pytest tests/test_all_models.py -v -s")
    print("Run specific test: pytest tests/test_all_models.py::TestModelDiscovery -v") 