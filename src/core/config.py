import os
from functools import lru_cache
from typing import Any, Dict, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Values are loaded from environment variables or .env file.
    Environment variables take precedence over .env file values.
    """

    # API Keys
    openai_api_key: str = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    midjourney_api_key: str = Field(
        default_factory=lambda: os.environ.get("MIDJOURNEY_API_KEY", "")
    )

    # Application Settings
    app_name: str = Field(
        default="Image Generation API", description="Name of the application"
    )
    app_host: str = Field(
        default="0.0.0.0", description="Host to bind the application to"
    )
    app_port: int = Field(default=8000, description="Port to bind the application to")
    app_url: str = Field(
        default="https://your-site.com", description="Public URL of the application"
    )

    # Provider API configurations
    provider_configs: Dict[str, Dict[str, Any]] = Field(
        default={
            "dalle": {
                "api_url": "https://api.openai.com/v1/images/generations",
                "api_version": "v1",
                "default_model": "dall-e-3",
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 1,
            },
            "midjourney": {
                "api_url": "https://api.midjourney.com/v1/generations",
                "api_version": "v1",
                "timeout": 60,
                "max_retries": 3,
                "retry_delay": 2,
            },
        },
        description="Provider-specific configurations",
    )

    # Request configurations
    request_timeout: int = Field(
        default=30, description="Default request timeout in seconds"
    )  # noqa: E501
    max_retries: int = Field(default=3, description="Maximum number of request retries")
    retry_delay: int = Field(default=1, description="Delay between retries in seconds")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Self":
        """Validate that all required API keys are set."""
        missing_keys = []
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.midjourney_api_key:
            missing_keys.append("MIDJOURNEY_API_KEY")

        if missing_keys:
            raise ValueError(
                f"Missing required API keys: {', '.join(missing_keys)}. "
                "Please set them in your environment or .env file."
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def verify_api_keys() -> bool:
    """Verify that all required API keys are set."""
    try:
        get_settings()
        return True
    except ValueError:
        return False
