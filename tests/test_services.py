import pytest
from unittest.mock import patch

from src.services.image_generator import OpenRouterGenerator, get_generator
from src.models.enums import Provider

@pytest.mark.asyncio
async def test_get_generator():
    generator = get_generator(Provider.OPENROUTER.value)
    assert isinstance(generator, OpenRouterGenerator)

@pytest.mark.asyncio
async def test_openrouter_generator():
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json.return_value = {
            "data": [{"url": "http://example.com/image.png"}]
        }
        
        generator = OpenRouterGenerator()
        result = await generator.generate("test prompt")
        
        assert "data" in result
        assert result["data"][0]["url"] == "http://example.com/image.png" 
