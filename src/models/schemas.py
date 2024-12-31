from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel

from .enums import Provider, TaskStatus


class ImageRequest(BaseModel):
    prompt: str
    provider: Provider
    style: Optional[str] = None
    size: str = "1024x1024"
    additional_params: Optional[Dict[str, Any]] = None


class ImageResponse(BaseModel):
    task_id: str
    status: TaskStatus
    prompt: str
    provider: Provider
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
