import uuid
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.models.enums import TaskStatus
from src.models.schemas import ImageRequest as APIImageRequest
from src.models.schemas import ImageResponse
from src.services.image_generator import generate_image
from src.services.task_events import task_events
from src.storage.memory import storage

router = APIRouter()


async def process_image_request(task_id: str, request: APIImageRequest) -> None:
    """
    Processes image generation request asynchronously
    """
    task = storage.get_task(task_id)
    if not task:
        return

    task.status = TaskStatus.PROCESSING
    storage.save_task(task)
    await task_events.notify_task_started(task, request.webhook_url)

    try:
        result_url = await generate_image(
            prompt=request.prompt,
            provider=request.provider.value,
            size=request.size,
            **(request.additional_params or {}),
        )

        if result_url:
            task.result_url = result_url
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED
            task.error_message = "Failed to generate image: No URL returned"

        task.completed_at = datetime.now()

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.completed_at = datetime.now()

    storage.save_task(task)
    await task_events.notify_task_completed(task, request.webhook_url)


@router.post("/generate/image", response_model=ImageResponse)
async def generate_image_endpoint(
    request: APIImageRequest, background_tasks: BackgroundTasks
) -> ImageResponse:
    """
    Initiates an asynchronous image generation request

    Returns a task object that can be used to track the generation progress
    """
    task_id = str(uuid.uuid4())
    task = ImageResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        prompt=request.prompt,
        provider=request.provider,
        created_at=datetime.now(),
    )

    storage.save_task(task)
    await task_events.notify_task_created(task, request.webhook_url)
    background_tasks.add_task(process_image_request, task_id, request)
    return task


@router.get("/status/{task_id}", response_model=ImageResponse)
async def get_task_status(task_id: str) -> ImageResponse:
    """Gets the status of a specific task"""
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, str]:
    """Deletes a specific task"""
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if storage.delete_task(task_id):
        await task_events.notify_task_deleted(task, None)  # No webhook_url for delete
        return {"status": "deleted"}

    raise HTTPException(status_code=404, detail="Task not found")
