from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.config import get_settings

# Load environment variables from .env file
load_dotenv()

# Settings will raise ValueError if API keys are missing
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="API for generating images using various AI providers",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=True)
