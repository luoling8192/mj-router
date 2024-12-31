import os
from typing import Generator

import pytest

from src.core.config import Settings, get_settings


@pytest.fixture
def clean_env() -> Generator:
    """Remove all API keys from environment for testing."""
    original = {}
    keys = ["OPENAI_API_KEY", "MIDJOURNEY_API_KEY"]

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
        Settings()

    error_msg = str(exc_info.value)
    assert "Missing required API keys" in error_msg
    assert "OPENAI_API_KEY" in error_msg
    assert "MIDJOURNEY_API_KEY" in error_msg


def test_settings_validation_succeeds_with_keys(clean_env: None) -> None:
    """Test that Settings validation succeeds when all API keys are present."""
    os.environ.update(
        {
            "OPENAI_API_KEY": "test-key",
            "MIDJOURNEY_API_KEY": "test-key",
        }
    )

    settings = Settings()
    assert settings.openai_api_key == "test-key"
    assert settings.midjourney_api_key == "test-key"


def test_settings_defaults() -> None:
    """Test that Settings uses correct default values."""
    settings = get_settings()
    assert settings.app_name == "Image Generation API"
    assert settings.app_host == "0.0.0.0"
    assert settings.app_port == 8000
    assert settings.app_url == "https://your-site.com"
