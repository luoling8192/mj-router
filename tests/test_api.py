import logging

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.enums import Provider, TaskStatus
from tests.conftest import has_api_keys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = TestClient(app)


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_generate_image():
    logger.info("Starting image generation test")
    logger.debug("Preparing request payload")

    payload = {
        "prompt": "test prompt",
        "provider": Provider.DALLE.value,
        "size": "1024x1024",
    }
    logger.debug(f"Request payload: {payload}")

    logger.info("Sending POST request to /api/generate/image")
    response = client.post(
        "/api/generate/image",
        json=payload,
    )
    logger.debug(f"Response status code: {response.status_code}")

    assert response.status_code == 200
    data = response.json()
    logger.debug(f"Response data: {data}")

    assert data["status"] == TaskStatus.PENDING.value
    assert data["prompt"] == "test prompt"
    assert "task_id" in data
    logger.info("Image generation test completed successfully")


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_get_task_status():
    logger.info("Starting task status test")

    # First create a task
    logger.debug("Creating new task")
    response = client.post(
        "/api/generate/image",
        json={"prompt": "test prompt", "provider": Provider.DALLE.value},
    )
    task_data = response.json()
    logger.debug(f"Task creation response: {task_data}")

    task_id = task_data["task_id"]
    logger.info(f"Task created with ID: {task_id}")

    # Then get its status
    logger.debug(f"Checking status for task {task_id}")
    response = client.get(f"/api/status/{task_id}")
    logger.debug(f"Status check response code: {response.status_code}")

    assert response.status_code == 200
    data = response.json()
    logger.debug(f"Status check response data: {data}")

    assert data["task_id"] == task_id
    logger.info("Task status test completed successfully")
