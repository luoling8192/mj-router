import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from src.core.config import get_settings
from src.services.providers.base import (
    APIError,
    BaseProvider,
    ImageRequest,
    ImageUrl,
    JsonResponse,
    RequestConfig,
)
from src.services.utils.http import make_request


class MJTaskStatus(str, Enum):
    """Midjourney task status"""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "SUCCESS"
    FAILED = "FAILED"


@dataclass(frozen=True)
class MJAccount:
    """Represents a Midjourney account status"""

    id: str
    channel_id: str
    guild_id: str
    core_size: int
    queue_size: int
    timeout_minutes: int
    user_agent: str
    user_token: str
    enable: bool
    properties: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MJAccount":
        """Create MJAccount instance from API response"""
        return cls(
            id=data.get("id", ""),
            channel_id=data.get("channelId", ""),
            guild_id=data.get("guildId", ""),
            core_size=data.get("coreSize", 0),
            queue_size=data.get("queueSize", 0),
            timeout_minutes=data.get("timeoutMinutes", 0),
            user_agent=data.get("userAgent", ""),
            user_token=data.get("userToken", ""),
            enable=data.get("enable", False),
            properties=data.get("properties", {}),
        )


class MidjourneyProvider(BaseProvider):
    """Midjourney image generation provider"""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(settings.PROVIDER_CONFIGS["midjourney"])

    def create_request(self, request: ImageRequest) -> RequestConfig:
        """Creates Midjourney API request configuration"""
        additional_params = request.additional_params or {}

        payload = {
            "prompt": request.prompt,
            "base64Array": [],
            "notifyHook": "",
            "state": "",
            **additional_params,
        }

        return RequestConfig(
            url=f"{self.config['api_url']}/submit/imagine",
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout=self.config["timeout"],
            max_retries=self.config["max_retries"],
            retry_delay=self.config["retry_delay"],
        )

    def transform_response(self, response: JsonResponse) -> Optional[str]:
        """Extracts task ID from Midjourney submit response"""
        try:
            if response.get("code") != 1:
                raise APIError(
                    400, response.get("description", "Failed to submit task")
                )
            return str(response["result"])
        except (KeyError, TypeError):
            return None

    async def get_accounts(self) -> List[MJAccount]:
        """Fetch all Midjourney accounts status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['api_url']}/account/list"
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(response.status, error_text)

                    data = await response.json()
                    return [MJAccount.from_dict(account) for account in data]
        except aiohttp.ClientError as e:
            raise APIError(500, f"Network error while fetching accounts: {str(e)}")

    async def get_account(self, account_id: str) -> Optional[MJAccount]:
        """Fetch specific Midjourney account status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['api_url']}/account/{account_id}/fetch"
                ) as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(response.status, error_text)

                    data = await response.json()
                    return MJAccount.from_dict(data)
        except aiohttp.ClientError as e:
            raise APIError(
                500, f"Network error while fetching account {account_id}: {str(e)}"
            )

    async def _get_available_account(self) -> Optional[str]:
        """Get an available Midjourney account ID for task submission"""
        accounts = await self.get_accounts()
        if not accounts:
            return None

        available_accounts = [
            acc for acc in accounts if acc.enable and acc.queue_size < acc.core_size
        ]

        if not available_accounts:
            return None

        available_accounts.sort(
            key=lambda acc: acc.queue_size / acc.core_size
            if acc.core_size > 0
            else float("inf")
        )

        return available_accounts[0].id

    async def _poll_task(self, task_id: str) -> Optional[ImageUrl]:
        """Polls task status until completion or timeout"""
        max_attempts = self.config.get("poll_max_attempts", 30)
        delay = self.config.get("poll_interval", 10)

        for _ in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.config['api_url']}/task/{task_id}/fetch"
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise APIError(response.status, error_text)

                        data = await response.json()
                        status = data.get("status", "")

                        if status == MJTaskStatus.COMPLETED:
                            return data.get("imageUrl")
                        elif status == MJTaskStatus.FAILED:
                            raise APIError(400, data.get("failReason", "Task failed"))

                        await asyncio.sleep(delay)
            except aiohttp.ClientError as e:
                raise APIError(500, f"Network error while polling: {str(e)}")

        raise APIError(408, "Timeout waiting for task completion")

    async def generate(self, request: ImageRequest) -> Optional[ImageUrl]:
        """Generate image using Midjourney"""
        try:
            # Get available account if not specified
            account_id = (
                request.additional_params.get("account_id")
                if request.additional_params
                else None
            )
            if not account_id:
                account_id = await self._get_available_account()
                if not account_id:
                    raise APIError(503, "No available Midjourney accounts")
                request = request.with_params(
                    additional_params={"account_id": account_id}
                )

            # Submit task
            config = self.create_request(request)
            response = await make_request(config)
            task_id = self.transform_response(response)

            if not task_id:
                return None

            # Poll for result
            return await self._poll_task(task_id)
        except APIError as e:
            raise APIError(e.status_code, e.message)
