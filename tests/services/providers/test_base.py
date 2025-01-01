from typing import Any, Dict

from src.services.providers.base import (
    APIError,
    BaseProvider,
    ImageRequest,
    ImageUrl,
    JsonResponse,
    RequestConfig,
)


def test_image_request():
    """Test ImageRequest creation and parameter updates"""
    request = ImageRequest(prompt="test")
    assert request.prompt == "test"
    assert request.size == "1024x1024"  # Default value

    new_request = request.with_params(size="512x512")
    assert new_request.prompt == request.prompt
    assert new_request.size == "512x512"
    assert request.size == "1024x1024"  # Original unchanged

    # Test additional params
    request_with_params = request.with_params(additional_params={"key": "value"})
    assert request_with_params.additional_params == {"key": "value"}


def test_api_error():
    """Test APIError creation and message formatting"""
    error = APIError(400, "Bad Request")
    assert error.status_code == 400
    assert error.message == "Bad Request"
    assert str(error) == "API Error (400): Bad Request"


class MockProvider(BaseProvider):
    """Mock provider for testing BaseProvider functionality"""

    def create_request(self, request: ImageRequest) -> RequestConfig:
        return RequestConfig(
            url="http://test.com",
            headers={},
            payload={},
            timeout=30,
            max_retries=3,
            retry_delay=1,
        )

    def transform_response(self, response: JsonResponse) -> ImageUrl:
        return response.get("url", "")

    async def generate(self, request: ImageRequest) -> ImageUrl:
        return "http://test.com/image.jpg"


def test_base_provider():
    """Test BaseProvider functionality"""
    config: Dict[str, Any] = {"key": "value"}
    provider = MockProvider(config)

    assert provider.get_config() == config

    request = ImageRequest(prompt="test")
    config2 = provider.create_request(request)
    assert isinstance(config2, RequestConfig)
    assert config2.url == "http://test.com"
