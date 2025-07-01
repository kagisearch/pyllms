"""
Pytest configuration and fixtures for PyLLMs test suite.
"""

# Automatically load .env file if it exists (must be done before any other imports)
import os
from pathlib import Path

# Try to load .env file automatically, but don't break if dotenv isn't installed
try:
    from dotenv import load_dotenv
    
    # Look for .env file in project root (parent of tests directory)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        # Don't print during import to avoid cluttering output
        
except ImportError:
    # Silently continue if python-dotenv is not installed
    pass

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take time due to API calls)"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: marks tests that require API keys"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their characteristics."""
    for item in items:
        # Mark tests that make API calls as slow
        if "completion" in item.name.lower() or "async" in item.name.lower() or "stream" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that require API keys
        if "test_model_" in item.name or "test_async_" in item.name or "test_streaming_" in item.name:
            item.add_marker(pytest.mark.requires_api_key)


@pytest.fixture(scope="session")
def test_environment():
    """Fixture to check test environment setup."""
    env_info = {
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "has_google_key": bool(os.getenv("GOOGLE_API_KEY")),
        "has_groq_key": bool(os.getenv("GROQ_API_KEY")),
        "has_mistral_key": bool(os.getenv("MISTRAL_API_KEY")),
        "has_deepseek_key": bool(os.getenv("DEEPSEEK_API_KEY")),
        "has_cohere_key": bool(os.getenv("COHERE_API_KEY")),
        "total_api_keys": sum([
            bool(os.getenv("OPENAI_API_KEY")),
            bool(os.getenv("ANTHROPIC_API_KEY")),
            bool(os.getenv("GOOGLE_API_KEY")),
            bool(os.getenv("GROQ_API_KEY")),
            bool(os.getenv("MISTRAL_API_KEY")),
            bool(os.getenv("DEEPSEEK_API_KEY")),
            bool(os.getenv("COHERE_API_KEY")),
        ])
    }
    return env_info


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Auto-use fixture to set up test environment."""
    # Set conservative default timeouts to avoid hanging tests
    os.environ.setdefault("PYTEST_TIMEOUT", "30")
    yield
    # Cleanup if needed
    pass


def pytest_runtest_setup(item):
    """Setup for each test item."""
    # Skip tests if no API keys are available and test requires them
    if item.get_closest_marker("requires_api_key"):
        api_keys_available = any([
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GROQ_API_KEY"),
            os.getenv("MISTRAL_API_KEY"),
            os.getenv("DEEPSEEK_API_KEY"),
            os.getenv("COHERE_API_KEY"),
        ])
        if not api_keys_available:
            pytest.skip("No API keys available for testing")


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("\n" + "="*60)
    print("PyLLMs Test Suite")
    print("="*60)
    
    # Check for available API keys
    api_keys = {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "BedrockAnthropic": bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")),
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
        "Groq": bool(os.getenv("GROQ_API_KEY")),
        "Mistral": bool(os.getenv("MISTRAL_API_KEY")),
        "DeepSeek": bool(os.getenv("DEEPSEEK_API_KEY")),
        "Cohere": bool(os.getenv("COHERE_API_KEY")),
        "Together": bool(os.getenv("TOGETHER_API_KEY")),
        "OpenRouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "Reka": bool(os.getenv("REKA_API_KEY")),
        "AI21": bool(os.getenv("AI21_API_KEY")),
        "AlephAlpha": bool(os.getenv("ALEPHALPHA_API_KEY")),
        "HuggingfaceHub": bool(os.getenv("HUGGINFACEHUB_API_KEY")),
    }
    
    available_keys = [name for name, available in api_keys.items() if available]
    print(f"API Keys Available: {len(available_keys)}/{len(api_keys)}")
    
    if available_keys:
        print("Available providers:", ", ".join(available_keys))
    else:
        print("⚠️  No API keys found - only testing provider discovery and initialization")
    
    print("="*60)


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    print("\n" + "="*60)
    print("PyLLMs Test Suite Complete")
    print("="*60) 