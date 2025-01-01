from unittest.mock import patch

import pytest

from src.services.providers.base import ImageRequest
from src.services.providers.openai import OpenAIProvider


@pytest.fixture
def openai_provider():
    with patch("src.services.providers.openai.get_settings") as mock_settings:
        mock_settings.return_value.OPENAI_API_KEY = "test-key"
        mock_settings.return_value.PROVIDER_CONFIGS = {
            "dalle": {
                "api_url": "https://api.openai.com/v1/images/generations",
                "default_model": "dall-e-3",
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 1,
            }
        }
        yield OpenAIProvider()


def test_create_request(openai_provider):
    """Test DALL-E request configuration creation"""
    request = ImageRequest(prompt="test prompt", size="1024x1024")
    config = openai_provider.create_request(request)

    assert config.url == "https://api.openai.com/v1/images/generations"
    assert config.headers["Authorization"] == "Bearer test-key"
    assert config.payload["prompt"] == "test prompt"
    assert config.payload["model"] == "dall-e-3"
    assert config.payload["size"] == "1024x1024"


def test_transform_response(openai_provider):
    """Test DALL-E response transformation"""
    response = {"data": [{"url": "https://example.com/image.png"}]}
    url = openai_provider.transform_response(response)
    assert url == "https://example.com/image.png"

    # Test invalid response
    assert openai_provider.transform_response({}) is None


@pytest.mark.asyncio
async def test_generate(openai_provider):
    """Test DALL-E image generation flow"""
    with patch("src.services.providers.openai.make_request") as mock_request:
        mock_request.return_value = {"data": [{"url": "https://example.com/image.png"}]}

        result = await openai_provider.generate(ImageRequest(prompt="test"))
        assert result == "https://example.com/image.png"
