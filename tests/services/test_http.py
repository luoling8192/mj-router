from unittest.mock import AsyncMock, patch

import pytest

from src.services.providers.base import APIError, RequestConfig
from src.services.utils.http import make_request


@pytest.mark.asyncio
async def test_make_request_success():
    """Test successful HTTP request"""
    config = RequestConfig(
        url="http://test.com",
        headers={},
        payload={},
        timeout=1,
        max_retries=1,
        retry_delay=0,
    )

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"success": True}

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await make_request(config)
        assert result == {"success": True}


@pytest.mark.asyncio
async def test_make_request_error():
    """Test HTTP request with error"""
    config = RequestConfig(
        url="http://test.com",
        headers={},
        payload={},
        timeout=1,
        max_retries=1,
        retry_delay=0,
    )

    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text.return_value = "Server Error"

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(APIError) as exc_info:
            await make_request(config)

        assert exc_info.value.status_code == 500
        assert "Server Error" in str(exc_info.value)
