from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes import router
from core.config import get_settings

load_dotenv()

app = FastAPI(title=get_settings().app_name)
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=True)
