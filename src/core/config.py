from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


def load_yaml_config() -> Dict[str, Any]:
    """Load and merge YAML configuration files"""
    # Get the project root directory
    root_dir = Path(__file__).parent.parent.parent

    # Load base config
    base_config_path = root_dir / "config.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    with open(base_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load local config if exists
    local_config_path = root_dir / "config.local.yaml"
    if local_config_path.exists():
        with open(local_config_path, encoding="utf-8") as f:
            local_config = yaml.safe_load(f)
            deep_merge(config, local_config)

    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override dict into base dict"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


class AppConfig(BaseSettings):
    """Application configuration"""

    name: str = Field(default="Image Generation API")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    url: str = Field(default="https://your-site.com")

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        extra="ignore",
    )


class WebhookConfig(BaseSettings):
    """Webhook configuration"""

    timeout: int = Field(default=10)
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=1)
    default_url: str = Field(default="")

    model_config = SettingsConfigDict(
        env_prefix="WEBHOOK_",
        extra="ignore",
    )


class ApiKeys(BaseSettings):
    """API keys configuration"""

    openai: str = Field(default="", alias="OPENAI_API_KEY")

    model_config = SettingsConfigDict(
        extra="ignore",
    )


class RequestConfig(BaseSettings):
    """Global request configuration"""

    timeout: int = Field(default=30)
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=1)

    model_config = SettingsConfigDict(
        env_prefix="REQUEST_",
        extra="ignore",
    )


class Settings(BaseSettings):
    """Application settings"""

    app: AppConfig
    api_keys: ApiKeys
    providers: Dict[str, Dict[str, Any]]
    request: RequestConfig
    webhook: WebhookConfig

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls) -> "Settings":
        """Create settings from YAML configuration and environment variables"""
        # Load YAML config first
        config = load_yaml_config()

        # Create settings with YAML config as defaults, but API keys from env
        settings = cls(
            app=AppConfig(**config["app"]),
            api_keys=ApiKeys(),  # API keys from environment variables
            providers=config["providers"],
            request=RequestConfig(**config["request"]),
            webhook=WebhookConfig(**config.get("webhook", {})),
        )

        return settings

    @field_validator("api_keys")
    def validate_api_keys(cls, v: ApiKeys) -> ApiKeys:
        """Validate required API keys"""
        if not v.openai:
            raise ValueError(
                "Missing required API key: OPENAI_API_KEY. "
                "Please set it in .env file or environment variables."
            )
        return v

    @property
    def OPENAI_API_KEY(self) -> str:
        """Compatibility property for OpenAI API key"""
        return self.api_keys.openai

    @property
    def PROVIDER_CONFIGS(self) -> Dict[str, Dict[str, Any]]:
        """Compatibility property for provider configs"""
        return self.providers

    @property
    def MAX_RETRIES(self) -> int:
        """Compatibility property for max retries"""
        return self.request.max_retries

    @property
    def RETRY_DELAY(self) -> int:
        """Compatibility property for retry delay"""
        return self.request.retry_delay


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    try:
        return Settings.from_yaml()
    except Exception as e:
        import logging

        logging.error(f"Failed to load settings: {str(e)}")
        raise


def verify_api_keys() -> bool:
    """Verify API keys are valid"""
    try:
        settings = get_settings()
        return bool(settings.api_keys.openai)
    except Exception:
        return False
