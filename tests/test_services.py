import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.models.enums import Provider
from src.services.image_generator import (
    APIError,
    ImageGenerator,
    ImageRequest,
    RequestConfig,
    image_generator,
    make_request,
)
from tests.conftest import has_api_keys


def test_image_request():
    request = ImageRequest(prompt="test")
    new_request = request.with_params(size="512x512")

    assert request.prompt == new_request.prompt
    assert new_request.size == "512x512"
    assert request.size == "1024x1024"  # Original unchanged


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_dalle_request_config():
    """Test DALL-E request configuration"""
    generator = ImageGenerator()
    request = ImageRequest(prompt="test prompt", size="1024x1024")
    config = generator._create_dalle_request(request)

    assert isinstance(config, RequestConfig)
    assert config.url == "https://api.openai.com/v1/images/generations"
    assert "Authorization" in config.headers
    assert config.payload["prompt"] == "test prompt"
    assert config.payload["model"] == "dall-e-3"


async def test_get_generator():
    dalle_gen = image_generator.generate_with_provider
    midjourney_gen = image_generator.generate_with_provider

    assert callable(dalle_gen)
    assert callable(midjourney_gen)

    with pytest.raises(ValueError):
        await image_generator.generate_with_provider("invalid_provider", "test")


@pytest.mark.asyncio
async def test_dalle_image_generation():
    """Test DALL-E image generation flow"""
    mock_response = {"data": [{"url": "https://example.com/image.png"}]}

    with patch(
        "src.services.image_generator.make_request",
        AsyncMock(return_value=mock_response),
    ):
        result = await image_generator.generate_with_provider(
            Provider.DALLE.value, "test prompt", size="1024x1024"
        )

        assert result == "https://example.com/image.png"


def test_response_transformers():
    """Test response transformers for different providers"""
    generator = ImageGenerator()

    # Test DALL-E response
    dalle_response = {"data": [{"url": "https://dalle.example.com/image.png"}]}
    assert (
        generator._transform_dalle_response(dalle_response)
        == "https://dalle.example.com/image.png"
    )

    # Test Midjourney response
    midjourney_response = {"image_url": "https://midjourney.example.com/image.png"}
    assert (
        generator._transform_midjourney_response(midjourney_response)
        == "https://midjourney.example.com/image.png"
    )


def test_invalid_response_handling():
    """Test handling of invalid response formats"""
    generator = ImageGenerator()
    invalid_response = {"wrong_key": "value"}

    assert generator._transform_dalle_response(invalid_response) is None
    assert generator._transform_midjourney_response(invalid_response) is None


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in image generation"""
    with patch(
        "src.services.image_generator.make_request",
        AsyncMock(side_effect=Exception("API Error")),
    ):
        with pytest.raises(Exception) as exc_info:
            await image_generator.generate_with_provider(
                Provider.DALLE.value, "test prompt"
            )

        assert str(exc_info.value) == "API Error"


@pytest.mark.asyncio
async def test_request_retry_logic() -> None:
    """Test that requests are retried on failure"""
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="Server Error")

    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_response

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_session

    config = RequestConfig(
        url="http://test.com",
        headers={},
        payload={},
        timeout=1,
        max_retries=2,
        retry_delay=0,
    )

    with patch("aiohttp.ClientSession", return_value=mock_client):
        with pytest.raises(APIError) as exc_info:
            await make_request(config)

        assert "Server Error" in str(exc_info.value)
        # Should be called max_retries times
        assert mock_client.__aenter__.call_count == 2


@pytest.mark.asyncio
async def test_request_timeout() -> None:
    """Test that requests respect timeout configuration"""

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(0.2)
        return AsyncMock(status=200, json=AsyncMock(return_value={}))

    mock_session = AsyncMock()
    mock_session.post = slow_response

    config = RequestConfig(
        url="http://test.com",
        headers={},
        payload={},
        timeout=1,  # Very short timeout
        max_retries=1,
        retry_delay=0,
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(APIError) as exc_info:
            await make_request(config)

        assert "Network error" in str(exc_info.value)
