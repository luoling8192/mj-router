import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, cast

import aiohttp
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_fixed

from src.core.config import get_settings
from src.models.enums import Provider

# Custom types for better type safety
ImageUrl = str
JsonResponse = Dict[str, Any]


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    model: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

    def with_params(self, **kwargs: Any) -> "ImageRequest":
        """Creates a new ImageRequest with updated parameters"""
        return ImageRequest(
            prompt=self.prompt,
            size=cast(str, kwargs.get("size", self.size)),
            quality=cast(str, kwargs.get("quality", self.quality)),
            model=cast(Optional[str], kwargs.get("model", self.model)),
            additional_params={
                **(self.additional_params or {}),
                **kwargs.get("additional_params", {}),
            },
        )


@dataclass(frozen=True)
class RequestConfig:
    url: str
    headers: Dict[str, str]
    payload: Dict[str, Any]
    timeout: int
    max_retries: int
    retry_delay: int


T = TypeVar("T")
RequestTransformer = Callable[[ImageRequest], RequestConfig]
ResponseTransformer = Callable[[JsonResponse], Optional[ImageUrl]]


class APIError(Exception):
    """Custom exception for API-related errors"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error ({status_code}): {message}")


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for each provider including request and response handling"""

    request_transformer: RequestTransformer
    response_transformer: ResponseTransformer


async def make_request(config: RequestConfig) -> JsonResponse:
    """Makes HTTP request with retry logic and configurable timeout"""

    @retry(
        stop=stop_after_attempt(config.max_retries), wait=wait_fixed(config.retry_delay)
    )
    async def _make_request_with_retry() -> JsonResponse:
        try:
            timeout = aiohttp.ClientTimeout(total=config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    config.url, headers=config.headers, json=config.payload
                ) as response:
                    if response.status == 200:
                        json_response: JsonResponse = await response.json()
                        return json_response
                    error_text = await response.text()
                    raise APIError(response.status, error_text)
        except aiohttp.ClientError as e:
            raise APIError(500, f"Network error: {str(e)}")  # noqa: B904

    return await _make_request_with_retry()


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for routing requests to different providers"""

    default_provider: str
    fallback_provider: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 1
    fallback_errors: tuple = (APIError, ValueError)  # Errors that trigger fallback


class ImageRouter:
    """Routes image generation requests to appropriate providers"""

    def __init__(self, config: RouterConfig, generator: "ImageGenerator") -> None:
        self.config = config
        self.generator = generator

    async def route_request(
        self, prompt: str, provider: Optional[str] = None, **kwargs: Any
    ) -> Optional[ImageUrl]:
        """Routes the request to specified provider with fallback support"""
        current_provider = provider or self.config.default_provider

        try:
            return await self.generator.generate_with_provider(
                current_provider, prompt, **kwargs
            )
        except self.config.fallback_errors as e:
            if (
                self.config.fallback_provider
                and current_provider != self.config.fallback_provider
            ):
                # Log the error before attempting fallback
                logging.warning(
                    f"Provider {current_provider} failed: {str(e)}. "
                    f"Falling back to {self.config.fallback_provider}"
                )
                return await self.generator.generate_with_provider(
                    self.config.fallback_provider, prompt, **kwargs
                )
            raise


# Add new data classes for account management
@dataclass(frozen=True)
class MJAccount:
    """Represents a Midjourney account status"""
    id: str
    channel_id: str
    guild_id: str
    core_size: int
    queue_size: int
    timeout_minutes: int
    user_agent: str
    user_token: str
    enable: bool
    properties: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MJAccount":
        """Create MJAccount instance from API response"""
        return cls(
            id=data.get("id", ""),
            channel_id=data.get("channelId", ""),
            guild_id=data.get("guildId", ""),
            core_size=data.get("coreSize", 0),
            queue_size=data.get("queueSize", 0),
            timeout_minutes=data.get("timeoutMinutes", 0),
            user_agent=data.get("userAgent", ""),
            user_token=data.get("userToken", ""),
            enable=data.get("enable", False),
            properties=data.get("properties", {})
        )


class ImageGenerator:
    """Handles image generation requests for different providers"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider_configs = self._initialize_configs()
        self.router = ImageRouter(
            RouterConfig(
                default_provider=Provider.DALLE.value,
                fallback_provider=Provider.MIDJOURNEY.value,
                max_retries=self.settings.MAX_RETRIES,
                retry_delay=self.settings.RETRY_DELAY,
            ),
            self,
        )

    def _initialize_configs(self) -> Dict[str, ProviderConfig]:
        """Initialize provider configurations"""
        return {
            Provider.DALLE.value: ProviderConfig(
                request_transformer=self._create_dalle_request,
                response_transformer=self._transform_dalle_response,
            ),
            Provider.MIDJOURNEY.value: ProviderConfig(
                request_transformer=self._create_midjourney_request,
                response_transformer=self._transform_midjourney_response,
            ),
        }

    def _create_dalle_request(self, request: ImageRequest) -> RequestConfig:
        """Creates DALL-E API request configuration"""
        provider_config = self.settings.PROVIDER_CONFIGS["dalle"]
        return RequestConfig(
            url=provider_config["api_url"],
            headers={
                "Authorization": f"Bearer {self.settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            payload={
                "prompt": request.prompt,
                "size": request.size,
                "quality": request.quality,
                "model": provider_config["default_model"],
                "n": 1,
            },
            timeout=provider_config["timeout"],
            max_retries=provider_config["max_retries"],
            retry_delay=provider_config["retry_delay"],
        )

    def _create_midjourney_request(self, request: ImageRequest) -> RequestConfig:
        """Creates Midjourney API request configuration"""
        provider_config = self.settings.PROVIDER_CONFIGS["midjourney"]

        # Get additional parameters or empty dict
        additional_params = request.additional_params or {}

        # If account_id is not provided, it will be selected automatically during task submission
        payload = {
            "prompt": request.prompt,
            "base64Array": [],
            "notifyHook": "",
            "state": "",
            **additional_params
        }

        return RequestConfig(
            url=f"{provider_config['api_url']}/submit/imagine",
            headers={
                "Authorization": f"Bearer {self.settings.MIDJOURNEY_API_KEY}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout=provider_config["timeout"],
            max_retries=provider_config["max_retries"],
            retry_delay=provider_config["retry_delay"],
        )

    async def _poll_midjourney_task(self, task_id: str, max_attempts: int = 30, delay: int = 10) -> Optional[str]:
        """Polls Midjourney task status until completion or timeout"""
        provider_config = self.settings.PROVIDER_CONFIGS["midjourney"]

        for _ in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{provider_config['api_url']}/task/{task_id}/fetch",
                        headers={"Authorization": f"Bearer {self.settings.MIDJOURNEY_API_KEY}"}
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise APIError(response.status, error_text)

                        data = await response.json()
                        status = data.get("status", "")

                        if status == MJTaskStatus.COMPLETED:
                            return data.get("imageUrl")
                        elif status == MJTaskStatus.FAILED:
                            raise APIError(400, data.get("failReason", "Task failed"))

                        # Task still in progress, wait before next attempt
                        await asyncio.sleep(delay)
            except aiohttp.ClientError as e:
                raise APIError(500, f"Network error while polling: {str(e)}")

        raise APIError(408, "Timeout waiting for Midjourney task completion")

    @staticmethod
    def _transform_dalle_response(response: JsonResponse) -> Optional[ImageUrl]:
        """Extracts image URL from DALL-E response"""
        try:
            return cast(str, response["data"][0]["url"])
        except (KeyError, IndexError):
            return None

    @staticmethod
    def _transform_midjourney_response(response: JsonResponse) -> Optional[ImageUrl]:
        """Extracts task ID from Midjourney submit response"""
        try:
            if response.get("code") != 1:
                raise APIError(400, response.get("description", "Failed to submit task"))
            return str(response["result"])  # Return task ID as string
        except (KeyError, TypeError):
            return None

    async def generate_image(
        self, provider_config: ProviderConfig, prompt: str, **kwargs: Any
    ) -> Optional[ImageUrl]:
        """Generic image generation function that returns the image URL"""
        try:
            request = ImageRequest(prompt=prompt, **kwargs)
            config = provider_config.request_transformer(request)

            # For Midjourney, try to get an available account first
            if "midjourney" in config.url:
                account_id = kwargs.get("account_id")
                if not account_id:
                    account_id = await self._get_available_mj_account()
                    if not account_id:
                        raise APIError(503, "No available Midjourney accounts")
                    config.payload["account_id"] = account_id

            response = await make_request(config)

            # For Midjourney, we need to poll for the final result
            if "midjourney" in config.url:
                task_id = provider_config.response_transformer(response)
                if task_id:
                    return await self._poll_midjourney_task(task_id)
                return None

            return provider_config.response_transformer(response)
        except APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

    async def generate_with_provider(
        self, provider: str, prompt: str, **kwargs: Any
    ) -> Optional[ImageUrl]:
        """Generates image using specified provider"""
        if provider not in self.provider_configs:
            raise ValueError(f"Unsupported provider: {provider}")

        config = self.provider_configs[provider]
        return await self.generate_image(config, prompt, **kwargs)

    async def route_request(
        self, prompt: str, provider: Optional[str] = None, **kwargs: Any
    ) -> Optional[ImageUrl]:
        """Public method to route requests through the router"""
        return await self.router.route_request(prompt, provider, **kwargs)

    async def get_mj_accounts(self) -> list[MJAccount]:
        """Fetch all Midjourney accounts status"""
        provider_config = self.settings.PROVIDER_CONFIGS["midjourney"]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{provider_config['api_url']}/account/list",
                    headers={"Authorization": f"Bearer {self.settings.MIDJOURNEY_API_KEY}"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(response.status, error_text)

                    data = await response.json()
                    return [MJAccount.from_dict(account) for account in data]
        except aiohttp.ClientError as e:
            raise APIError(500, f"Network error while fetching accounts: {str(e)}")

    async def get_mj_account(self, account_id: str) -> Optional[MJAccount]:
        """Fetch specific Midjourney account status"""
        provider_config = self.settings.PROVIDER_CONFIGS["midjourney"]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{provider_config['api_url']}/account/{account_id}/fetch",
                    headers={"Authorization": f"Bearer {self.settings.MIDJOURNEY_API_KEY}"}
                ) as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(response.status, error_text)

                    data = await response.json()
                    return MJAccount.from_dict(data)
        except aiohttp.ClientError as e:
            raise APIError(500, f"Network error while fetching account {account_id}: {str(e)}")

    async def _get_available_mj_account(self) -> Optional[str]:
        """Get an available Midjourney account ID for task submission"""
        accounts = await self.get_mj_accounts()

        # Filter enabled accounts with available capacity
        available_accounts = [
            acc for acc in accounts
            if acc.enable and acc.queue_size < acc.core_size
        ]

        # Sort by current load (queue_size/core_size ratio)
        available_accounts.sort(
            key=lambda acc: acc.queue_size / acc.core_size if acc.core_size > 0 else float('inf')
        )

        return available_accounts[0].id if available_accounts else None


# Global instance
image_generator = ImageGenerator()


async def generate_image(
    prompt: str, provider: Optional[str] = None, **kwargs: Any
) -> Optional[ImageUrl]:
    """
    Global async function for generating images with automatic routing

    Args:
        prompt: The image generation prompt
        provider: Optional provider override (DALLE or MIDJOURNEY)
        **kwargs: Additional provider-specific parameters

    Returns:
        Optional[ImageUrl]: The generated image URL if successful

    Raises:
        HTTPException: If image generation fails
    """
    return await image_generator.route_request(prompt, provider, **kwargs)


# Add new constants
class MJTaskStatus(str, Enum):
    """Midjourney task status"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "SUCCESS"
    FAILED = "FAILED"
