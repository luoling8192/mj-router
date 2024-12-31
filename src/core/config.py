import os
from functools import lru_cache

from pydantic import ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings.

    Values are loaded from environment variables or .env file.
    Environment variables take precedence over .env file values.
    """

    # API Keys
    openai_api_key: str = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    openrouter_api_key: str = Field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
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

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        """Validate that all required API keys are set."""
        missing_keys = []
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.openrouter_api_key:
            missing_keys.append("OPENROUTER_API_KEY")
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
