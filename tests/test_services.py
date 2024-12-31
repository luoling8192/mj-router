import pytest
from src.services.image_generator import (
    ImageRequest,
    RequestConfig,
    create_dalle_request,
    create_openrouter_request,
    get_generator,
)
from src.models.enums import Provider

def test_image_request():
    request = ImageRequest(prompt="test")
    new_request = request.with_params(size="512x512")
    
    assert request.prompt == new_request.prompt
    assert new_request.size == "512x512"
    assert request.size == "1024x1024"  # Original unchanged

def test_dalle_request_config():
    request = ImageRequest(prompt="test prompt", size="1024x1024")
    config = create_dalle_request(request)
    
    assert isinstance(config, RequestConfig)
    assert config.url == "https://api.openai.com/v1/images/generations"
    assert "Authorization" in config.headers
    assert config.payload["prompt"] == "test prompt"
    assert config.payload["model"] == "dall-e-3"

def test_openrouter_request_config():
    request = ImageRequest(
        prompt="test prompt",
        model="openai/dall-e-3"
    )
    config = create_openrouter_request(request)
    
    assert isinstance(config, RequestConfig)
    assert config.url == "https://openrouter.ai/api/v1/images/generations"
    assert "Authorization" in config.headers
    assert config.payload["prompt"] == "test prompt"
    assert config.payload["model"] == "openai/dall-e-3"

@pytest.mark.asyncio
async def test_get_generator():
    dalle_gen = get_generator(Provider.DALLE.value)
    openrouter_gen = get_generator(Provider.OPENROUTER.value)
    
    assert callable(dalle_gen)
    assert callable(openrouter_gen)
    
    with pytest.raises(ValueError):
        get_generator("invalid_provider") 
