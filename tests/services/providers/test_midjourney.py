from unittest.mock import AsyncMock, patch

import pytest

from src.services.providers.base import APIError, ImageRequest
from src.services.providers.midjourney import (
    MidjourneyProvider,
)


@pytest.fixture
def midjourney_provider():
    with patch("src.services.providers.midjourney.get_settings") as mock_settings:
        mock_settings.return_value.PROVIDER_CONFIGS = {
            "midjourney": {
                "api_url": "http://mj-api.com",
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 1,
                "poll_max_attempts": 2,
                "poll_interval": 1,
            }
        }
        yield MidjourneyProvider()


@pytest.fixture
def mock_account():
    return {
        "id": "test-1",
        "channelId": "channel-1",
        "guildId": "guild-1",
        "coreSize": 3,
        "queueSize": 1,
        "timeoutMinutes": 30,
        "userAgent": "test-agent",
        "userToken": "test-token",
        "enable": True,
        "properties": {},
    }


def test_create_request(midjourney_provider):
    """Test Midjourney request configuration creation"""
    request = ImageRequest(prompt="test", additional_params={"account_id": "test-1"})
    config = midjourney_provider.create_request(request)

    assert config.url == "http://mj-api.com/submit/imagine"
    assert "Content-Type" in config.headers
    assert "Authorization" not in config.headers  # 确保没有 Authorization header
    assert config.payload["prompt"] == "test"
    assert config.payload["account_id"] == "test-1"


@pytest.mark.asyncio
async def test_get_accounts_empty(midjourney_provider):
    """Test fetching Midjourney accounts when no accounts available"""
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        accounts = await midjourney_provider.get_accounts()
        assert len(accounts) == 0

        # Test getting available account when no accounts exist
        account_id = await midjourney_provider._get_available_account()
        assert account_id is None


@pytest.mark.asyncio
async def test_get_accounts_all_busy(midjourney_provider):
    """Test fetching Midjourney accounts when all accounts are busy"""
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json.return_value = [
            {
                "id": "test-1",
                "coreSize": 2,
                "queueSize": 2,  # Full capacity
                "enable": True,
            },
            {
                "id": "test-2",
                "coreSize": 1,
                "queueSize": 1,  # Full capacity
                "enable": True,
            },
        ]
        mock_get.return_value = mock_response

        # Test getting available account when all accounts are busy
        account_id = await midjourney_provider._get_available_account()
        assert account_id is None


@pytest.mark.asyncio
async def test_generate_no_accounts(midjourney_provider):
    """Test image generation when no accounts are available"""
    with patch.object(
        midjourney_provider, "_get_available_account"
    ) as mock_get_account:
        mock_get_account.return_value = None

        with pytest.raises(APIError) as exc_info:
            await midjourney_provider.generate(ImageRequest(prompt="test"))

        assert exc_info.value.status_code == 503
        assert "No available Midjourney accounts" in str(exc_info.value.message)
