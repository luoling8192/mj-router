import asyncio
import logging

import httpx

from src.core.config import get_settings
from src.models.schemas import ImageResponse

logger = logging.getLogger(__name__)


async def send_webhook(webhook_url: str, task: ImageResponse) -> bool:
    """
    Send webhook notification for task status update

    Returns:
        bool: True if webhook was sent successfully, False otherwise
    """
    logger.info(
        "Sending webhook notification - Task ID: %s, Status: %s",
        task.task_id,
        task.status,
    )
    logger.debug("Webhook URL: %s", webhook_url)

    if not webhook_url:
        settings = get_settings()
        webhook_url = settings.webhook.default_url
        if not webhook_url:
            logger.info("No webhook URL configured, skipping notification")
            return False
        logger.info("Using default webhook URL")
        logger.debug("Default webhook URL: %s", webhook_url)

    settings = get_settings()
    retry_count = 0
    logger.debug(
        "Webhook configuration - Timeout: %ds, Max Retries: %d, Retry Delay: %ds",
        settings.webhook.timeout,
        settings.webhook.max_retries,
        settings.webhook.retry_delay,
    )

    while retry_count < settings.webhook.max_retries:
        try:
            logger.debug(
                "Sending webhook notification (attempt %d/%d)",
                retry_count + 1,
                settings.webhook.max_retries,
            )
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=task.model_dump(),
                    timeout=settings.webhook.timeout,
                )
                response.raise_for_status()
                logger.info(
                    "Webhook notification sent successfully - Task ID: %s",
                    task.task_id,
                )
                logger.debug("Response status code: %d", response.status_code)
                return True
        except httpx.TimeoutException:
            retry_count += 1
            logger.error(
                "Webhook request timed out - Task ID: %s (attempt %d/%d)",
                task.task_id,
                retry_count,
                settings.webhook.max_retries,
            )
            if retry_count >= settings.webhook.max_retries:
                logger.error(
                    "Webhook notification failed due to timeout - Task ID: %s",
                    task.task_id,
                )
                return False
            logger.info(
                "Retrying webhook request in %ds - Task ID: %s",
                settings.webhook.retry_delay,
                task.task_id,
            )
            await asyncio.sleep(settings.webhook.retry_delay)
        except httpx.HTTPError as e:
            retry_count += 1
            logger.error(
                "Webhook request failed with HTTP error - Task ID: %s, Error: %s (attempt %d/%d)",
                task.task_id,
                str(e),
                retry_count,
                settings.webhook.max_retries,
            )
            if retry_count >= settings.webhook.max_retries:
                logger.error(
                    "Webhook notification failed due to HTTP error - Task ID: %s",
                    task.task_id,
                )
                return False
            logger.info(
                "Retrying webhook request in %ds - Task ID: %s",
                settings.webhook.retry_delay,
                task.task_id,
            )
            await asyncio.sleep(settings.webhook.retry_delay)
        except Exception as e:
            retry_count += 1
            logger.error(
                "Unexpected error during webhook request - Task ID: %s, Error: %s (attempt %d/%d)",
                task.task_id,
                str(e),
                retry_count,
                settings.webhook.max_retries,
            )
            if retry_count >= settings.webhook.max_retries:
                logger.error(
                    "Webhook notification failed due to unexpected error - Task ID: %s",
                    task.task_id,
                )
                return False
            logger.info(
                "Retrying webhook request in %ds - Task ID: %s",
                settings.webhook.retry_delay,
                task.task_id,
            )
            await asyncio.sleep(settings.webhook.retry_delay)

    return False
