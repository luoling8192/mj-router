import logging
from dataclasses import dataclass
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


class ImageGenerator:
    """Handles image generation requests for different providers"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider_configs = self._initialize_configs()
        self.router = ImageRouter(
            RouterConfig(
                default_provider=Provider.DALLE.value,
                fallback_provider=Provider.MIDJOURNEY.value,
                max_retries=self.settings.max_retries,
                retry_delay=self.settings.retry_delay,
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
        provider_config = self.settings.provider_configs["dalle"]
        return RequestConfig(
            url=provider_config["api_url"],
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
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
        provider_config = self.settings.provider_configs["midjourney"]
        return RequestConfig(
            url=provider_config["api_url"],
            headers={
                "Authorization": f"Bearer {self.settings.midjourney_api_key}",
                "Content-Type": "application/json",
            },
            payload={"prompt": request.prompt, **(request.additional_params or {})},
            timeout=provider_config["timeout"],
            max_retries=provider_config["max_retries"],
            retry_delay=provider_config["retry_delay"],
        )

    @staticmethod
    def _transform_dalle_response(response: JsonResponse) -> Optional[ImageUrl]:
        """Extracts image URL from DALL-E response"""
        try:
            return cast(str, response["data"][0]["url"])
        except (KeyError, IndexError):
            return None

    @staticmethod
    def _transform_midjourney_response(response: JsonResponse) -> Optional[ImageUrl]:
        """Extracts image URL from Midjourney response"""
        try:
            return cast(Optional[str], response.get("image_url"))
        except (KeyError, IndexError):
            return None

    async def generate_image(
        self, provider_config: ProviderConfig, prompt: str, **kwargs: Any
    ) -> Optional[ImageUrl]:
        """Generic image generation function that returns the image URL"""
        try:
            request = ImageRequest(prompt=prompt, **kwargs)
            config = provider_config.request_transformer(request)
            response = await make_request(config)
            return provider_config.response_transformer(response)
        except APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)  # noqa: B904

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
