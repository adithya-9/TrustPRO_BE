"""TrustPRO API application.

Run:  uvicorn app.main:app --port 8000
Docs: http://localhost:8000/api/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
log = logging.getLogger("trustpro")


def _download_missing_models() -> None:
    s = settings
    needed = [s.models_dir / "face_detection_yunet_2023mar.onnx", s.models_dir / "arcface_w600k_r50.onnx",
              s.models_dir / s.environment_model]
    if not s.auto_download_models or all(p.exists() for p in needed):
        return
    log.info("Model files missing - downloading them (first start can take a few minutes)")
    from scripts.download_models import main as download_models

    download_models()


def _warm_models() -> None:
    """Download missing model files, then load models in the background so the first request
    does not pay the start-up cost."""
    try:
        _download_missing_models()
    except BaseException:  # noqa: BLE001 - SystemExit from a checksum failure must not kill the server
        log.exception("Model download failed; analysis needs the files in %s", settings.models_dir)
    from app.ai.face import get_face_engine
    from app.ai.ocr import get_ocr_engine
    from app.pipeline.environment import get_environment_model

    loaders = (("face", get_face_engine), ("ocr", get_ocr_engine), ("environment", get_environment_model))
    for name, loader in loaders:
        try:
            loader()
            log.info("Model ready: %s", name)
        except Exception:
            log.exception("Model failed to load: %s", name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from app.db.session import engine
    from app.pipeline.jobs import jobs
    from app.pipeline.live import registry as live_registry

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Database connection OK")
        resumed = jobs.recover()
        if resumed:
            log.info("Re-queued %s unfinished report(s)", resumed)
    except Exception:
        log.exception("Database not reachable at start-up; requests will report it until it is")
    warm = ThreadPoolExecutor(max_workers=1, thread_name_prefix="warmup")
    warm.submit(_warm_models)
    yield
    live_registry.shutdown()
    jobs.shutdown()
    warm.shutdown(wait=False)


def create_app() -> FastAPI:
    app = FastAPI(
        title="TrustPRO API",
        description="TrustPRO - candidate assessments: profile, ID verification, recorded assessment and a "
                    "server-side report (ID / face comparison and environmental anomalies).",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )
    register_error_handlers(app)

    from app.api.routes import auth, candidate, interviews, media, recruiter

    app.include_router(auth.router)
    app.include_router(candidate.router)
    app.include_router(interviews.router)
    app.include_router(media.router)
    app.include_router(recruiter.router)

    @app.get("/api/health", tags=["System"], summary="Liveness and database check")
    def health() -> dict:
        from app.db.session import engine

        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False
        return {"status": "ok" if db_ok else "degraded", "database": db_ok}

    return app


app = create_app()
