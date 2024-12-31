from fastapi.testclient import TestClient

from src.main import app
from src.models.enums import Provider, TaskStatus

client = TestClient(app)


def test_generate_image():
    response = client.post(
        "/api/v1/generate/image",
        json={
            "prompt": "test prompt",
            "provider": Provider.OPENROUTER.value,
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "test prompt"
    assert data["provider"] == Provider.OPENROUTER.value
    assert data["status"] == TaskStatus.PENDING.value
    assert "task_id" in data


def test_get_task_status_not_found():
    response = client.get("/api/v1/status/nonexistent-task")
    assert response.status_code == 404
