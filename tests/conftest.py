from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.core.config import ApiKeys, AppConfig, RequestConfig, Settings, get_settings
from src.main import app


def has_api_keys() -> bool:
    """Check if required API keys are configured"""
    settings = get_settings()
    return bool(settings.api_keys.openai)


@pytest.fixture
def settings() -> Generator[Settings, None, None]:
    """Provide test settings"""
    test_settings = Settings(
        app=AppConfig(
            name="Test API",
            host="localhost",
            port=8000,
            url="http://test.com",
        ),
        api_keys=ApiKeys(openai="test-openai-key"),
        providers={
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
        request=RequestConfig(
            timeout=30,
            max_retries=3,
            retry_delay=1,
        ),
    )

    with patch("src.core.config.get_settings", return_value=test_settings):
        yield test_settings


@pytest.fixture
def client(settings) -> Generator[TestClient, None, None]:
    """Provide test client"""
    with TestClient(app) as test_client:
        yield test_client
