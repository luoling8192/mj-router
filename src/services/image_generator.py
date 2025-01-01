from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import HTTPException

from src.core.config import get_settings
from src.models.enums import Provider
from src.services.providers.base import APIError, ImageProvider, ImageRequest, ImageUrl
from src.services.providers.midjourney import MidjourneyProvider
from src.services.providers.openai import OpenAIProvider


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for routing requests to different providers"""

    default_provider: str
    fallback_provider: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 1
    fallback_errors: tuple = (APIError, ValueError)


class ImageRouter:
    """Routes image generation requests to appropriate providers"""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._providers: Dict[str, ImageProvider] = {}

    def register_provider(self, name: str, provider: ImageProvider) -> None:
        """Register a new provider"""
        self._providers[name] = provider

    async def route_request(
        self, prompt: str, provider: Optional[str] = None, **kwargs: Any
    ) -> Optional[ImageUrl]:
        """Routes the request to specified provider with fallback support"""
        current_provider = provider or self.config.default_provider

        if current_provider not in self._providers:
            raise ValueError(f"Unsupported provider: {current_provider}")

        try:
            request = ImageRequest(prompt=prompt, **kwargs)
            return await self._providers[current_provider].generate(request)
        except self.config.fallback_errors as e:
            if (
                self.config.fallback_provider
                and current_provider != self.config.fallback_provider
                and self.config.fallback_provider in self._providers
            ):
                import logging

                logging.warning(
                    f"Provider {current_provider} failed: {str(e)}. "
                    f"Falling back to {self.config.fallback_provider}"
                )
                request = ImageRequest(prompt=prompt, **kwargs)
                return await self._providers[self.config.fallback_provider].generate(
                    request
                )
            raise


class ImageGenerator:
    """Handles image generation requests for different providers"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.router = self._initialize_router()

    def _initialize_router(self) -> ImageRouter:
        """Initialize router with providers"""
        router = ImageRouter(
            RouterConfig(
                default_provider=Provider.DALLE.value,
                fallback_provider=Provider.MIDJOURNEY.value,
                max_retries=self.settings.MAX_RETRIES,
                retry_delay=self.settings.RETRY_DELAY,
            )
        )

        # Register providers
        router.register_provider(Provider.DALLE.value, OpenAIProvider())
        router.register_provider(Provider.MIDJOURNEY.value, MidjourneyProvider())

        return router

    async def generate(
        self, prompt: str, provider: Optional[str] = None, **kwargs: Any
    ) -> Optional[ImageUrl]:
        """Generate image with automatic routing"""
        try:
            return await self.router.route_request(prompt, provider, **kwargs)
        except APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)


# Global instance
image_generator = ImageGenerator()


async def generate_image(
    prompt: str, provider: Optional[str] = None, **kwargs: Any
) -> Optional[ImageUrl]:
    """Global async function for generating images with automatic routing"""
    return await image_generator.generate(prompt, provider, **kwargs)
