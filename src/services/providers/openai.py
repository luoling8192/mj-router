from typing import Optional

from src.core.config import get_settings
from src.services.providers.base import (
    APIError,
    BaseProvider,
    ImageRequest,
    ImageUrl,
    JsonResponse,
    RequestConfig,
)
from src.services.utils.http import make_request


class OpenAIProvider(BaseProvider):
    """OpenAI/DALL-E image generation provider"""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(settings.PROVIDER_CONFIGS["dalle"])
        self.api_key = settings.OPENAI_API_KEY

    def create_request(self, request: ImageRequest) -> RequestConfig:
        """Creates DALL-E API request configuration"""
        return RequestConfig(
            url=self.config["api_url"],
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "prompt": request.prompt,
                "size": request.size,
                "quality": request.quality,
                "model": self.config["default_model"],
                "n": 1,
            },
            timeout=self.config["timeout"],
            max_retries=self.config["max_retries"],
            retry_delay=self.config["retry_delay"],
        )

    def transform_response(self, response: JsonResponse) -> Optional[ImageUrl]:
        """Extracts image URL from DALL-E response"""
        try:
            return str(response["data"][0]["url"])
        except (KeyError, IndexError):
            return None

    async def generate(self, request: ImageRequest) -> Optional[ImageUrl]:
        """Generate image using DALL-E"""
        try:
            config = self.create_request(request)
            response = await make_request(config)
            return self.transform_response(response)
        except APIError as e:
            raise APIError(e.status_code, e.message)
