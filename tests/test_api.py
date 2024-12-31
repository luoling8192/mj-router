from fastapi.testclient import TestClient

from src.main import app
from src.models.enums import Provider, TaskStatus

client = TestClient(app)


def test_generate_image():
    response = client.post(
        "/api/generate/image",
        json={
            "prompt": "test prompt",
            "provider": Provider.DALLE.value,
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == TaskStatus.PENDING.value
    assert data["prompt"] == "test prompt"
    assert "task_id" in data


def test_get_task_status():
    # First create a task
    response = client.post(
        "/api/generate/image",
        json={"prompt": "test prompt", "provider": Provider.DALLE.value},
    )
    task_id = response.json()["task_id"]

    # Then get its status
    response = client.get(f"/api/status/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
