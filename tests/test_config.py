import os
from typing import Generator

import pytest

from src.core.config import (
    ApiKeys,
    AppConfig,
    RequestConfig,
    Settings,
    WebhookConfig,
    get_settings,
)


@pytest.fixture
def clean_env() -> Generator:
    """Remove all API keys from environment for testing."""
    original = {}
    keys = ["OPENAI_API_KEY"]

    # Save and remove environment variables
    for key in keys:
        original[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]

    yield

    # Restore environment variables
    for key, value in original.items():
        if value is not None:
            os.environ[key] = value


def test_settings_validation_fails_without_keys(clean_env: None) -> None:
    """Test that Settings validation fails when API keys are missing."""
    with pytest.raises(ValueError) as exc_info:
        Settings(
            app=AppConfig(
                name="test", host="localhost", port=8000, url="http://test.com"
            ),
            api_keys=ApiKeys(OPENAI_API_KEY=""),
            providers={},
            request=RequestConfig(timeout=30, max_retries=3, retry_delay=1),
            webhook=WebhookConfig(
                timeout=10,
                max_retries=3,
                retry_delay=1,
                default_url="http://localhost:8888/webhook",
            ),
        )

    error_msg = str(exc_info.value)
    assert "Missing required API key: OPENAI_API_KEY" in error_msg


def test_settings_validation_succeeds_with_keys(clean_env: None) -> None:
    """Test that Settings validation succeeds when all API keys are present."""
    os.environ["OPENAI_API_KEY"] = "test-key"

    settings = Settings(
        app=AppConfig(
            name="test",
            host="localhost",
            port=8000,
            url="http://test.com",
        ),
        api_keys=ApiKeys(OPENAI_API_KEY="test-key"),
        providers={},
        request=RequestConfig(timeout=30, max_retries=3, retry_delay=1),
        webhook=WebhookConfig(
            timeout=10,
            max_retries=3,
            retry_delay=1,
            default_url="http://localhost:8888/webhook",
        ),
    )
    assert settings.OPENAI_API_KEY == "test-key"


def test_settings_defaults() -> None:
    """Test that Settings uses correct default values."""
    settings = get_settings()
    assert settings.app.name == "Image Generation API"
    assert settings.app.host == "0.0.0.0"
    assert settings.app.port == 8000
    assert settings.app.url == "https://your-site.com"


def test_provider_config_defaults() -> None:
    """Test that provider configurations have correct default values"""
    settings = Settings(
        app=AppConfig(
            name="test",
            host="localhost",
            port=8000,
            url="http://test.com",
        ),
        api_keys=ApiKeys(OPENAI_API_KEY="test-key"),
        providers={
            "dalle": {
                "api_url": "https://api.openai.com/v1/images/generations",
                "default_model": "dall-e-3",
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 1,
            }
        },
        request=RequestConfig(timeout=30, max_retries=3, retry_delay=1),
        webhook=WebhookConfig(
            timeout=10,
            max_retries=3,
            retry_delay=1,
            default_url="http://localhost:8888/webhook",
        ),
    )

    # Test DALL-E config
    dalle_config = settings.PROVIDER_CONFIGS["dalle"]
    assert dalle_config["api_url"] == "https://api.openai.com/v1/images/generations"
    assert dalle_config["default_model"] == "dall-e-3"
    assert dalle_config["timeout"] == 30
    assert dalle_config["max_retries"] == 3


def test_custom_provider_config(clean_env: None) -> None:
    """Test that provider configurations can be overridden"""
    settings = Settings(
        app=AppConfig(
            name="test",
            host="localhost",
            port=8000,
            url="http://test.com",
        ),
        api_keys=ApiKeys(OPENAI_API_KEY="test-key"),
        providers={
            "dalle": {
                "api_url": "https://api.openai.com/v1/images/generations",
                "default_model": "dall-e-3",
                "timeout": 45,
                "max_retries": 5,
                "retry_delay": 1,
            }
        },
        request=RequestConfig(timeout=30, max_retries=3, retry_delay=1),
        webhook=WebhookConfig(
            timeout=10,
            max_retries=3,
            retry_delay=1,
            default_url="http://localhost:8888/webhook",
        ),
    )
    dalle_config = settings.PROVIDER_CONFIGS["dalle"]
    assert dalle_config["timeout"] == 45
    assert dalle_config["max_retries"] == 5
    # Other values should remain at defaults
    assert dalle_config["api_url"] == "https://api.openai.com/v1/images/generations"
