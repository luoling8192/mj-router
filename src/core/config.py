from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str
    midjourney_api_key: str
    app_name: str = "Image Generation API"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()