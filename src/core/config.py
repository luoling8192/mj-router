import json
import os
from functools import lru_cache
from typing import Any, Dict, Self

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_env_config() -> Dict[str, Any]:
    """Load and parse environment configuration"""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Get all environment variables
    env_vars = dict(os.environ)

    # Parse JSON provider configs if present
    if provider_configs := env_vars.get("PROVIDER_CONFIGS"):
        try:
            env_vars["provider_configs"] = json.loads(provider_configs)
        except json.JSONDecodeError:
            pass

    return env_vars


class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key",
    )
    MIDJOURNEY_API_KEY: str = Field(
        default="",
        description="Midjourney API key",
    )

    # Application Settings
    APP_NAME: str = Field(
        default="Image Generation API",
        description="Name of the application",
    )
    APP_HOST: str = Field(
        default="0.0.0.0",
        description="Host to bind the application to",
    )
    APP_PORT: int = Field(
        default=8000,
        description="Port to bind the application to",
    )
    APP_URL: str = Field(
        default="https://your-site.com",
        description="Public URL of the application",
    )

    # Provider API configurations with type hints
    PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = Field(
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
                "api_url": "http://localhost:8080",
                "timeout": 60,
                "max_retries": 3,
                "retry_delay": 2,
                "poll_timeout": 300,
                "poll_interval": 10,
            },
        },
    )

    # Request configurations
    REQUEST_TIMEOUT: int = Field(default=30)
    MAX_RETRIES: int = Field(default=3)
    RETRY_DELAY: int = Field(default=1)

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_nested_delimiter="__",
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment configuration"""
        return cls(**load_env_config())

    @model_validator(mode="after")
    def validate_api_keys(self) -> Self:
        """Validate API keys"""
        missing_keys = []

        # Validate OpenAI API key
        if not self.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")

        # Validate Midjourney API key
        if not self.MIDJOURNEY_API_KEY:
            missing_keys.append("MIDJOURNEY_API_KEY")

        if missing_keys:
            raise ValueError(
                f"Missing required API keys: {', '.join(missing_keys)}. "
                "Please set them in your environment or .env file."
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    try:
        return Settings.from_env()
    except Exception as e:
        import logging

        logging.error(f"Failed to load settings: {str(e)}")
        raise


def verify_api_keys() -> bool:
    """Verify API keys are valid"""
    try:
        settings = get_settings()
        return bool(settings.OPENAI_API_KEY and settings.MIDJOURNEY_API_KEY)
    except Exception:
        return False
