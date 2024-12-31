from typing import Dict, Protocol

import aiohttp
from fastapi import HTTPException

from core.config import get_settings


class ImageGeneratorProtocol(Protocol):
    async def generate(self, prompt: str, **kwargs) -> Dict: ...


class OpenRouterGenerator:
    async def generate(self, prompt: str, model: str = "openai/dall-e-3") -> Dict:
        settings = get_settings()
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": settings.app_name,
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "prompt": prompt,
            "quality": "standard",
            "size": "1024x1024",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/images/generations",
                headers=headers,
                json=data,
            ) as response:
                if response.status == 200:
                    return await response.json()
                raise HTTPException(
                    status_code=response.status, detail=await response.text()
                )


class MidjourneyGenerator:
    async def generate(self, prompt: str) -> Dict:
        # 实现 Midjourney 的调用逻辑
        raise NotImplementedError("Midjourney generation not implemented")


def get_generator(provider: str) -> ImageGeneratorProtocol:
    generators = {
        "openrouter": OpenRouterGenerator(),
        "midjourney": MidjourneyGenerator(),
    }
    return generators[provider]
