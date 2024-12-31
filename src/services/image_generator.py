from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

import aiohttp
from fastapi import HTTPException

from src.core.config import get_settings
from src.models.enums import Provider


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    model: str | None = None
    additional_params: Dict[str, Any] | None = None

    def with_params(self, **kwargs) -> "ImageRequest":
        """Creates a new ImageRequest with updated parameters"""
        return ImageRequest(
            prompt=self.prompt,
            size=kwargs.get("size", self.size),
            quality=kwargs.get("quality", self.quality),
            model=kwargs.get("model", self.model),
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


T = TypeVar("T")
RequestTransformer = Callable[[ImageRequest], RequestConfig]
ResponseTransformer = Callable[[Dict], T]


async def make_request(config: RequestConfig) -> Dict:
    """Makes HTTP request and handles errors"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            config.url, headers=config.headers, json=config.payload
        ) as response:
            if response.status == 200:
                return await response.json()
            raise HTTPException(
                status_code=response.status, detail=await response.text()
            )


def create_dalle_request(request: ImageRequest) -> RequestConfig:
    """Creates DALL-E API request configuration"""
    settings = get_settings()
    return RequestConfig(
        url="https://api.openai.com/v1/images/generations",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        payload={
            "prompt": request.prompt,
            "size": request.size,
            "quality": request.quality,
            "model": "dall-e-3",
            "n": 1,
        },
    )


def create_midjourney_request(request: ImageRequest) -> RequestConfig:
    """Creates Midjourney API request configuration"""
    settings = get_settings()
    return RequestConfig(
        url="https://api.midjourney.com/v1/generations",
        headers={
            "Authorization": f"Bearer {settings.midjourney_api_key}",
            "Content-Type": "application/json",
        },
        payload={"prompt": request.prompt, **(request.additional_params or {})},
    )


class ResponseTransformer(Protocol):  # noqa: F811
    """Protocol for response transformation functions"""

    def __call__(self, response: Dict[str, Any]) -> Optional[str]: ...


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for each provider including request and response handling"""

    request_transformer: RequestTransformer
    response_transformer: ResponseTransformer


def transform_dalle_response(response: Dict[str, Any]) -> Optional[str]:
    """Extracts image URL from DALL-E response"""
    try:
        return response["data"][0]["url"]
    except (KeyError, IndexError):
        return None


def transform_midjourney_response(response: Dict[str, Any]) -> Optional[str]:
    """Extracts image URL from Midjourney response"""
    try:
        # Adjust this based on actual Midjourney API response structure
        return response.get("image_url")
    except (KeyError, IndexError):
        return None


# Update provider configurations
PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    Provider.DALLE.value: ProviderConfig(
        request_transformer=create_dalle_request,
        response_transformer=transform_dalle_response,
    ),
    Provider.MIDJOURNEY.value: ProviderConfig(
        request_transformer=create_midjourney_request,
        response_transformer=transform_midjourney_response,
    ),
}


async def generate_image(
    provider_config: ProviderConfig, prompt: str, **kwargs
) -> Optional[str]:
    """Generic image generation function that returns the image URL"""
    request = ImageRequest(prompt=prompt, **kwargs)
    config = provider_config.request_transformer(request)
    response = await make_request(config)
    return provider_config.response_transformer(response)


def get_generator(provider: str) -> Callable[[str, Dict[str, Any]], Optional[str]]:
    """Returns appropriate generator function based on provider"""
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}")

    config = PROVIDER_CONFIGS[provider]
    return lambda prompt, **kwargs: generate_image(config, prompt, **kwargs)
