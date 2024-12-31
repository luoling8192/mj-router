from typing import Dict, Optional

from src.models.schemas import ImageResponse


class MemoryStorage:
    def __init__(self):
        self._tasks: Dict[str, ImageResponse] = {}

    def get_task(self, task_id: str) -> Optional[ImageResponse]:
        return self._tasks.get(task_id)

    def save_task(self, task: ImageResponse) -> None:
        self._tasks[task.task_id] = task

    def delete_task(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False


# 全局单例
storage = MemoryStorage()
