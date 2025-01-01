from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from src.models.enums import Provider, TaskStatus


class ImageRequest(BaseModel):
    prompt: str
    provider: Provider
    size: str = "1024x1024"
    additional_params: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "A beautiful sunset over mountains",
                "provider": "dalle",
                "size": "1024x1024",
                "additional_params": {"quality": "standard"},
                "webhook_url": "https://api.example.com/webhook",
            }
        }
    )


class ImageResponse(BaseModel):
    task_id: str
    status: TaskStatus
    prompt: str
    provider: Provider
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "pending",
                "prompt": "A beautiful sunset over mountains",
                "provider": "dalle",
                "created_at": "2024-01-01T00:00:00Z",
                "completed_at": None,
                "result_url": None,
                "error_message": None,
            }
        }
    )
