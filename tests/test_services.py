import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.models.enums import Provider
from src.services.image_generator import (
    APIError,
    ImageGenerator,
    ImageRequest,
    MJAccount,
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


@pytest.fixture
def mock_mj_account_response():
    """Mock Midjourney account response"""
    return {
        "id": "test-account-1",
        "channelId": "channel-1",
        "guildId": "guild-1",
        "coreSize": 3,
        "queueSize": 1,
        "timeoutMinutes": 30,
        "userAgent": "test-agent",
        "userToken": "test-token",
        "enable": True,
        "properties": {"key": "value"},
    }


@pytest.fixture
def mock_mj_accounts_response():
    """Mock Midjourney accounts list response"""
    return [
        {
            "id": "test-account-1",
            "channelId": "channel-1",
            "guildId": "guild-1",
            "coreSize": 3,
            "queueSize": 1,
            "timeoutMinutes": 30,
            "userAgent": "test-agent",
            "userToken": "test-token",
            "enable": True,
            "properties": {},
        },
        {
            "id": "test-account-2",
            "channelId": "channel-2",
            "guildId": "guild-2",
            "coreSize": 2,
            "queueSize": 2,
            "timeoutMinutes": 30,
            "userAgent": "test-agent",
            "userToken": "test-token",
            "enable": True,
            "properties": {},
        },
    ]


class TestMJAccount:
    """Test MJAccount data class"""

    def test_from_dict(self, mock_mj_account_response):
        """Test creating MJAccount from dictionary"""
        account = MJAccount.from_dict(mock_mj_account_response)

        assert account.id == "test-account-1"
        assert account.channel_id == "channel-1"
        assert account.guild_id == "guild-1"
        assert account.core_size == 3
        assert account.queue_size == 1
        assert account.timeout_minutes == 30
        assert account.user_agent == "test-agent"
        assert account.user_token == "test-token"
        assert account.enable is True
        assert account.properties == {"key": "value"}

    def test_from_dict_with_missing_fields(self):
        """Test creating MJAccount with missing fields"""
        account = MJAccount.from_dict({})

        assert account.id == ""
        assert account.channel_id == ""
        assert account.guild_id == ""
        assert account.core_size == 0
        assert account.queue_size == 0
        assert account.timeout_minutes == 0
        assert account.user_agent == ""
        assert account.user_token == ""
        assert account.enable is False
        assert account.properties == {}


class TestImageGenerator:
    """Test ImageGenerator class"""

    @pytest.mark.asyncio
    async def test_get_mj_accounts(self, image_generator, mock_mj_accounts_response):
        """Test fetching Midjourney accounts list"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.json.return_value = mock_mj_accounts_response
            mock_get.return_value = mock_response

            accounts = await image_generator.get_mj_accounts()

            assert len(accounts) == 2
            assert all(isinstance(acc, MJAccount) for acc in accounts)
            assert accounts[0].id == "test-account-1"
            assert accounts[1].id == "test-account-2"

    @pytest.mark.asyncio
    async def test_get_mj_accounts_error(self, image_generator):
        """Test fetching Midjourney accounts list with error"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__.return_value = mock_response
            mock_response.text.return_value = "Internal Server Error"
            mock_get.return_value = mock_response

            with pytest.raises(APIError) as exc_info:
                await image_generator.get_mj_accounts()

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_mj_account(self, image_generator, mock_mj_account_response):
        """Test fetching specific Midjourney account"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.json.return_value = mock_mj_account_response
            mock_get.return_value = mock_response

            account = await image_generator.get_mj_account("test-account-1")

            assert isinstance(account, MJAccount)
            assert account.id == "test-account-1"
            assert account.core_size == 3
            assert account.queue_size == 1

    @pytest.mark.asyncio
    async def test_get_mj_account_not_found(self, image_generator):
        """Test fetching non-existent Midjourney account"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.__aenter__.return_value = mock_response
            mock_get.return_value = mock_response

            account = await image_generator.get_mj_account("non-existent")
            assert account is None

    @pytest.mark.asyncio
    async def test_get_available_mj_account(
        self, image_generator, mock_mj_accounts_response
    ):
        """Test getting available Midjourney account"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.json.return_value = mock_mj_accounts_response
            mock_get.return_value = mock_response

            account_id = await image_generator._get_available_mj_account()
            assert (
                account_id == "test-account-1"
            )  # Should select account with lower load

    @pytest.mark.asyncio
    async def test_get_available_mj_account_no_available(self, image_generator):
        """Test getting available Midjourney account when none available"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            # All accounts are at capacity or disabled
            mock_response.json.return_value = [
                {"id": "test-account-1", "coreSize": 2, "queueSize": 2, "enable": True},
                {
                    "id": "test-account-2",
                    "coreSize": 3,
                    "queueSize": 1,
                    "enable": False,
                },
            ]
            mock_get.return_value = mock_response

            account_id = await image_generator._get_available_mj_account()
            assert account_id is None

    @pytest.mark.asyncio
    async def test_generate_image_with_auto_account_selection(self, image_generator):
        """Test image generation with automatic account selection"""
        with (
            patch("src.services.image_generator.make_request") as mock_make_request,
            patch.object(
                image_generator, "_get_available_mj_account"
            ) as mock_get_account,
            patch.object(image_generator, "_poll_midjourney_task") as mock_poll,
        ):
            mock_get_account.return_value = "test-account-1"
            mock_make_request.return_value = {"code": 1, "result": "task-123"}
            mock_poll.return_value = "http://example.com/image.jpg"

            result = await image_generator.generate_with_provider(
                "midjourney", "test prompt"
            )

            assert result == "http://example.com/image.jpg"
            mock_get_account.assert_called_once()
            mock_make_request.assert_called_once()
            mock_poll.assert_called_once_with("task-123")

    @pytest.mark.asyncio
    async def test_generate_image_no_available_accounts(self, image_generator):
        """Test image generation when no accounts are available"""
        with patch.object(
            image_generator, "_get_available_mj_account"
        ) as mock_get_account:
            mock_get_account.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await image_generator.generate_with_provider(
                    "midjourney", "test prompt"
                )

            assert exc_info.value.status_code == 503
            assert "No available Midjourney accounts" in str(exc_info.value.detail)
