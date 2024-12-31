import pytest
from fastapi.testclient import TestClient
import os

@pytest.fixture(autouse=True)
def env_setup():
    """Set up test environment variables."""
    os.environ["OPENROUTER_API_KEY"] = "test_key"
    os.environ["MIDJOURNEY_API_KEY"] = "test_key"
    yield
    del os.environ["OPENROUTER_API_KEY"]
    del os.environ["MIDJOURNEY_API_KEY"] 
