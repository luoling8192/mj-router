import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


# 状态枚举
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# 模型提供商枚举
class Provider(str, Enum):
    DALLE = "dalle"
    MIDJOURNEY = "midjourney"
    OPENROUTER = "openrouter"


# 请求模型
class ImageRequest(BaseModel):
    prompt: str
    provider: Provider
    style: Optional[str] = None
    size: str = "1024x1024"
    additional_params: Optional[Dict[str, Any]] = None


# 响应模型
class ImageResponse(BaseModel):
    task_id: str
    status: TaskStatus
    prompt: str
    provider: Provider
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None


# 内存存储任务状态（实际应用中应该使用数据库）
tasks: Dict[str, ImageResponse] = {}


class ImageGenerator:
    def __init__(self):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.midjourney_key = os.getenv("MIDJOURNEY_API_KEY")

    async def generate_openrouter(
        self, prompt: str, model: str = "openai/dall-e-3"
    ) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "Your-App-Name",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "prompt": prompt,
            "quality": "standard",
            "size": "1024x1024",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/images/generations",
                headers=headers,
                json=data,
            ) as response:
                if response.status == 200:
                    return await response.json()
                raise HTTPException(
                    status_code=response.status, detail=await response.text()
                )

    async def generate_midjourney(self, prompt: str) -> Dict:
        # 实现 Midjourney 的调用逻辑
        pass


async def process_image_request(task_id: str, request: ImageRequest):
    generator = ImageGenerator()
    tasks[task_id].status = TaskStatus.PROCESSING

    try:
        if request.provider == Provider.OPENROUTER:
            result = await generator.generate_openrouter(request.prompt)
            tasks[task_id].result_url = result["data"][0]["url"]
        elif request.provider == Provider.MIDJOURNEY:
            result = await generator.generate_midjourney(request.prompt)
            tasks[task_id].result_url = result["image_url"]

        tasks[task_id].status = TaskStatus.COMPLETED
        tasks[task_id].completed_at = datetime.now()

    except Exception as e:
        tasks[task_id].status = TaskStatus.FAILED
        tasks[task_id].error_message = str(e)
        tasks[task_id].completed_at = datetime.now()


@app.post("/generate/image", response_model=ImageResponse)
async def generate_image(request: ImageRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    # 创建新任务
    task = ImageResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        prompt=request.prompt,
        provider=request.provider,
        created_at=datetime.now(),
    )
    tasks[task_id] = task

    # 在后台处理请求
    background_tasks.add_task(process_image_request, task_id, request)

    return task


@app.get("/status/{task_id}", response_model=ImageResponse)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


# 清理工具（实际应用中应该有更完善的清理机制）
@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    if task_id in tasks:
        del tasks[task_id]
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
