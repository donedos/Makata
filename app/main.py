from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.api.routes import router

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("makata")
except Exception:
    _VERSION = "0.1.0"

config.ensure_dirs()

app = FastAPI(title="Makata", description="Zero-shot voice cloning TTS (Tagalog/English)", version=_VERSION)
app.include_router(router)

FRONTEND = Path(__file__).resolve().parent / "frontend"


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/audio", StaticFiles(directory=str(config.OUTPUT_DIR)), name="audio")
