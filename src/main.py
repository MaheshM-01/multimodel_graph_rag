"""Application main entrypoint."""

import uvicorn
from src.api.app import create_app
from src.config.settings import get_settings

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG or settings.APP_ENV == "development",
    )
