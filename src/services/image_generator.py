from typing import Dict, Protocol, Any, Callable, TypeVar
from dataclasses import dataclass
from functools import partial

import aiohttp
from fastapi import HTTPException

from core.config import get_settings
from models.enums import Provider


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    model: str | None = None
    additional_params: Dict[str, Any] = None

    def with_params(self, **kwargs) -> 'ImageRequest':
        """Creates a new ImageRequest with updated parameters"""
        return ImageRequest(
            prompt=self.prompt,
            size=kwargs.get('size', self.size),
            quality=kwargs.get('quality', self.quality),
            model=kwargs.get('model', self.model),
            additional_params={
                **(self.additional_params or {}),
                **kwargs.get('additional_params', {})
            }
        )


@dataclass(frozen=True)
class RequestConfig:
    url: str
    headers: Dict[str, str]
    payload: Dict[str, Any]


T = TypeVar('T')
RequestTransformer = Callable[[ImageRequest], RequestConfig]
ResponseTransformer = Callable[[Dict], T]


async def make_request(config: RequestConfig) -> Dict:
    """Makes HTTP request and handles errors"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            config.url,
            headers=config.headers,
            json=config.payload
        ) as response:
            if response.status == 200:
                return await response.json()
            raise HTTPException(
                status_code=response.status,
                detail=await response.text()
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
        }
    )


def create_openrouter_request(request: ImageRequest) -> RequestConfig:
    """Creates OpenRouter API request configuration"""
    settings = get_settings()
    return RequestConfig(
        url="https://openrouter.ai/api/v1/images/generations",
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
            "Content-Type": "application/json",
        },
        payload={
            "model": request.model or "openai/dall-e-3",
            "prompt": request.prompt,
            "quality": request.quality,
            "size": request.size,
        }
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
        payload={
            "prompt": request.prompt,
            **(request.additional_params or {})
        }
    )


async def generate_image(
    request_transformer: RequestTransformer,
    prompt: str,
    **kwargs
) -> Dict:
    """Generic image generation function"""
    request = ImageRequest(prompt=prompt, **kwargs)
    config = request_transformer(request)
    return await make_request(config)


# Create specialized generator functions
generate_dalle = partial(generate_image, create_dalle_request)
generate_openrouter = partial(generate_image, create_openrouter_request)
generate_midjourney = partial(generate_image, create_midjourney_request)


def get_generator(provider: str) -> Callable[[str, ...], Dict]:
    """Returns appropriate generator function based on provider"""
    generators = {
        Provider.DALLE.value: generate_dalle,
        Provider.OPENROUTER.value: generate_openrouter,
        Provider.MIDJOURNEY.value: generate_midjourney,
    }
    
    if provider not in generators:
        raise ValueError(f"Unsupported provider: {provider}")
        
    return generators[provider]
