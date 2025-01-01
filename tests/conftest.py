from typing import Generator

import pytest
from fastapi.testclient import TestClient

from src.core.config import Settings, get_settings
from src.main import app


def has_api_keys() -> bool:
    """Check if required API keys are configured"""
    settings = get_settings()
    return bool(settings.OPENAI_API_KEY)


@pytest.fixture
def settings() -> Generator[Settings, None, None]:
    """Provide test settings"""
    test_settings = Settings(
        OPENAI_API_KEY="test-openai-key",
        PROVIDER_CONFIGS={
            "dalle": {
                "api_url": "https://api.openai.com/v1/images/generations",
                "default_model": "dall-e-3",
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 1,
            },
            "midjourney": {
                "api_url": "http://localhost:8080",
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 1,
                "poll_max_attempts": 2,
                "poll_interval": 1,
            },
        },
    )
    yield test_settings


@pytest.fixture
def client(settings) -> Generator[TestClient, None, None]:
    """Provide test client"""
    with TestClient(app) as test_client:
        yield test_client
