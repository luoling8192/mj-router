import uuid
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from models.enums import TaskStatus
from models.schemas import ImageRequest as APIImageRequest
from models.schemas import ImageResponse
from services.image_generator import get_generator
from storage.memory import storage

router = APIRouter()


async def process_image_request(task_id: str, request: APIImageRequest) -> None:
    """
    Processes image generation request asynchronously

    Uses functional generators to handle different providers while maintaining
    consistent task state management
    """
    task = storage.get_task(task_id)
    if not task:
        return

    task.status = TaskStatus.PROCESSING
    storage.save_task(task)

    try:
        # Get the appropriate generator function for the provider
        generator = get_generator(request.provider.value)

        # Call the generator with the prompt and additional parameters
        result = await generator(
            prompt=request.prompt,
            size=request.size,
            **(request.additional_params or {}),
        )

        # Extract URL based on provider response format
        task.result_url = (
            result["data"][0]["url"]
            if request.provider.value == "openrouter"
            else result["image_url"]
        )
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.completed_at = datetime.now()

    storage.save_task(task)


@router.post("/generate/image", response_model=ImageResponse)
async def generate_image(
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
    background_tasks.add_task(process_image_request, task_id, request)
    return task


@router.get("/status/{task_id}", response_model=ImageResponse)
async def get_task_status(task_id: str) -> ImageResponse:
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, str]:
    if storage.delete_task(task_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Task not found")
