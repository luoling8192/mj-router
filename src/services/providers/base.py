from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

# Custom types for better type safety
ImageUrl = str
JsonResponse = Dict[str, Any]
T = TypeVar("T")


@dataclass(frozen=True)
class ImageRequest:
    """Base request model for image generation"""

    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    model: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

    def with_params(self, **kwargs: Any) -> "ImageRequest":
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
    """Configuration for making HTTP requests"""

    url: str
    headers: Dict[str, str]
    payload: Dict[str, Any]
    timeout: int
    max_retries: int
    retry_delay: int


class APIError(Exception):
    """Custom exception for API-related errors"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error ({status_code}): {message}")


class ImageProvider(Protocol):
    """Protocol defining the interface for image providers"""

    @abstractmethod
    async def generate(self, request: ImageRequest) -> Optional[ImageUrl]:
        """Generate an image from the given request"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get provider configuration"""
        pass


RequestTransformer = Callable[[ImageRequest], RequestConfig]
ResponseTransformer = Callable[[JsonResponse], Optional[ImageUrl]]


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for each provider including request and response handling"""

    request_transformer: RequestTransformer
    response_transformer: ResponseTransformer


class BaseProvider(ABC):
    """Base class for image providers with common functionality"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def create_request(self, request: ImageRequest) -> RequestConfig:
        """Create request configuration for the provider"""
        pass

    @abstractmethod
    def transform_response(self, response: JsonResponse) -> Optional[ImageUrl]:
        """Transform provider response to image URL"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get provider configuration"""
        return self.config
