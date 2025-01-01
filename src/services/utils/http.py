import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed

from src.services.providers.base import APIError, JsonResponse, RequestConfig


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def make_request(config: RequestConfig) -> JsonResponse:
    """Makes HTTP request with retry logic and configurable timeout"""
    try:
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                config.url, headers=config.headers, json=config.payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                error_text = await response.text()
                raise APIError(response.status, error_text)
    except aiohttp.ClientError as e:
        raise APIError(500, f"Network error: {str(e)}")
