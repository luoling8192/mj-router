from unittest.mock import AsyncMock, patch

import pytest

from src.models.enums import Provider
from src.services.image_generator import (
    PROVIDER_CONFIGS,
    ImageRequest,
    RequestConfig,
    create_dalle_request,
    get_generator,
    transform_dalle_response,
    transform_midjourney_response,
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
    request = ImageRequest(prompt="test prompt", size="1024x1024")
    config = create_dalle_request(request)

    assert isinstance(config, RequestConfig)
    assert config.url == "https://api.openai.com/v1/images/generations"
    assert "Authorization" in config.headers
    assert config.payload["prompt"] == "test prompt"
    assert config.payload["model"] == "dall-e-3"


def test_get_generator():
    dalle_gen = get_generator(Provider.DALLE.value)
    midjourney_gen = get_generator(Provider.MIDJOURNEY.value)

    assert callable(dalle_gen)
    assert callable(midjourney_gen)

    with pytest.raises(ValueError):
        get_generator("invalid_provider")


@pytest.mark.asyncio
async def test_dalle_image_generation():
    """Test DALL-E image generation flow"""
    mock_response = {"data": [{"url": "https://example.com/image.png"}]}

    with patch(
        "src.services.image_generator.make_request",
        AsyncMock(return_value=mock_response),
    ):
        generator = get_generator(Provider.DALLE.value)
        result = await generator("test prompt", size="1024x1024")

        assert result == "https://example.com/image.png"


def test_response_transformers():
    """Test response transformers for different providers"""
    # Test DALL-E response
    dalle_response = {"data": [{"url": "https://dalle.example.com/image.png"}]}
    assert (
        transform_dalle_response(dalle_response)
        == "https://dalle.example.com/image.png"
    )

    # Test Midjourney response
    midjourney_response = {"image_url": "https://midjourney.example.com/image.png"}
    assert (
        transform_midjourney_response(midjourney_response)
        == "https://midjourney.example.com/image.png"
    )


def test_invalid_response_handling():
    """Test handling of invalid response formats"""
    invalid_response = {"wrong_key": "value"}

    assert transform_dalle_response(invalid_response) is None
    assert transform_midjourney_response(invalid_response) is None


def test_provider_configs():
    """Test provider configuration setup"""
    for provider in Provider:
        assert provider.value in PROVIDER_CONFIGS
        config = PROVIDER_CONFIGS[provider.value]
        assert hasattr(config, "request_transformer")
        assert hasattr(config, "response_transformer")


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in image generation"""
    with patch(
        "src.services.image_generator.make_request",
        AsyncMock(side_effect=Exception("API Error")),
    ):
        generator = get_generator(Provider.DALLE.value)

        with pytest.raises(Exception) as exc_info:
            await generator("test prompt")

        assert str(exc_info.value) == "API Error"
