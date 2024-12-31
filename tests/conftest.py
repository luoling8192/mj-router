import logging
import os
from typing import Generator

import pytest

from src.core.config import Settings, get_settings


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def has_api_keys() -> bool:
    """Check if all required API keys are present and not test values"""
    settings = get_settings()
    return all(
        [
            settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "test_key",
            settings.MIDJOURNEY_API_KEY and settings.MIDJOURNEY_API_KEY != "test_key",
        ]
    )


@pytest.fixture(autouse=True)
def env_setup() -> Generator:
    """Set up test environment variables if they don't exist."""
    original_env = {}
    test_keys = {
        "OPENAI_API_KEY": "test_key",
        "MIDJOURNEY_API_KEY": "test_key",
    }

    # Save original environment and set test values
    for key, value in test_keys.items():
        original_env[key] = os.environ.get(key)
        if not os.environ.get(key):
            os.environ[key] = value

    yield

    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    caplog.set_level(logging.DEBUG)
