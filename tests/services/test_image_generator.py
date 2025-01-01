from unittest.mock import AsyncMock, patch

import pytest

from src.models.enums import Provider
from src.services.image_generator import ImageGenerator, ImageRouter, RouterConfig
from src.services.providers.base import APIError, ImageProvider, ImageRequest


class MockProvider(ImageProvider):
    """Mock provider for testing"""

    def __init__(self, return_value: str = "http://example.com/image.jpg"):
        self.return_value = return_value
        self.generate_called = False

    async def generate(self, request: ImageRequest) -> str:
        self.generate_called = True
        return self.return_value

    def get_config(self):
        return {}


@pytest.fixture
def router():
    """Create router with mock providers"""
    config = RouterConfig(
        default_provider=Provider.DALLE.value,
        fallback_provider=Provider.MIDJOURNEY.value,
    )
    router = ImageRouter(config)
    router.register_provider(Provider.DALLE.value, MockProvider())
    router.register_provider(Provider.MIDJOURNEY.value, MockProvider())
    return router


def test_router_config():
    """Test router configuration"""
    config = RouterConfig(
        default_provider="test", fallback_provider="fallback", max_retries=5
    )
    assert config.default_provider == "test"
    assert config.fallback_provider == "fallback"
    assert config.max_retries == 5


@pytest.mark.asyncio
async def test_router_default_provider(router):
    """Test routing to default provider"""
    result = await router.route_request("test prompt")
    assert result == "http://example.com/image.jpg"


@pytest.mark.asyncio
async def test_router_specific_provider(router):
    """Test routing to specific provider"""
    result = await router.route_request("test", provider=Provider.MIDJOURNEY.value)
    assert result == "http://example.com/image.jpg"


@pytest.mark.asyncio
async def test_router_fallback(router):
    """Test fallback when primary provider fails"""
    failing_provider = MockProvider()
    failing_provider.generate = AsyncMock(side_effect=APIError(500, "Error"))

    router._providers[Provider.DALLE.value] = failing_provider
    result = await router.route_request("test")

    assert result == "http://example.com/image.jpg"  # Fallback result


@pytest.mark.asyncio
async def test_router_invalid_provider(router):
    """Test handling of invalid provider"""
    with pytest.raises(ValueError):
        await router.route_request("test", provider="invalid")


@pytest.mark.asyncio
async def test_image_generator():
    """Test ImageGenerator initialization and generation"""
    with patch("src.services.image_generator.get_settings") as mock_settings:
        mock_settings.return_value.MAX_RETRIES = 3
        mock_settings.return_value.RETRY_DELAY = 1

        generator = ImageGenerator()
        assert isinstance(generator.router, ImageRouter)

        # Test generation
        with patch.object(generator.router, "route_request") as mock_route:
            mock_route.return_value = "http://example.com/image.jpg"

            result = await generator.generate("test prompt")
            assert result == "http://example.com/image.jpg"
