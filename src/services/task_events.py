import logging
from typing import Optional

from src.models.schemas import ImageResponse
from src.services.webhook import send_webhook

logger = logging.getLogger(__name__)


class TaskEventService:
    """Service for handling task events and notifications"""

    @staticmethod
    async def notify_task_created(
        task: ImageResponse, webhook_url: Optional[str] = None
    ) -> None:
        """Notify when a task is created"""
        logger.info(
            "Task created - ID: %s, Provider: %s",
            task.task_id,
            task.provider,
        )
        logger.debug("Task status: %s", task.status)

        if webhook_url:
            logger.debug("Sending task created notification to webhook")
            success = await send_webhook(webhook_url, task)
            if not success:
                logger.error(
                    "Failed to send task created notification - Task ID: %s",
                    task.task_id,
                )

    @staticmethod
    async def notify_task_started(
        task: ImageResponse, webhook_url: Optional[str] = None
    ) -> None:
        """Notify when a task starts processing"""
        logger.info(
            "Task started processing - ID: %s, Provider: %s",
            task.task_id,
            task.provider,
        )
        logger.debug("Task prompt: %s", task.prompt)

        if webhook_url:
            logger.debug("Sending task started notification to webhook")
            success = await send_webhook(webhook_url, task)
            if not success:
                logger.error(
                    "Failed to send task started notification - Task ID: %s",
                    task.task_id,
                )

    @staticmethod
    async def notify_task_completed(
        task: ImageResponse, webhook_url: Optional[str] = None
    ) -> None:
        """Notify when a task is completed (success or failure)"""
        if task.status == "completed":
            logger.info(
                "Task completed successfully - ID: %s",
                task.task_id,
            )
            logger.debug("Result URL: %s", task.result_url)
        else:
            logger.error(
                "Task failed - ID: %s, Error: %s",
                task.task_id,
                task.error_message,
            )

        if webhook_url:
            logger.debug("Sending task completion notification to webhook")
            success = await send_webhook(webhook_url, task)
            if not success:
                logger.error(
                    "Failed to send task completion notification - Task ID: %s",
                    task.task_id,
                )

    @staticmethod
    async def notify_task_deleted(
        task: ImageResponse, webhook_url: Optional[str] = None
    ) -> None:
        """Notify when a task is deleted"""
        logger.info(
            "Task deleted - ID: %s",
            task.task_id,
        )
        logger.debug("Final status: %s", task.status)

        if webhook_url:
            logger.debug("Sending task deletion notification to webhook")
            success = await send_webhook(webhook_url, task)
            if not success:
                logger.error(
                    "Failed to send task deletion notification - Task ID: %s",
                    task.task_id,
                )


# Create a singleton instance
task_events = TaskEventService()
