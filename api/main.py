"""RoomAI Production FastAPI Application Entrypoint."""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import detect, experimental, health, jobs, models, uploads
from api.schemas import ErrorDetail, ErrorResponse
from utils.logging_config import get_logger

logger = get_logger("RoomAI_API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: starts non-blocking background initialization for Model A3."""
    logger.info("Scheduling Model A3 background initialization at application startup...")
    detect.GLOBAL_MODEL_A3_ENGINE.start_background_init()
    try:
        yield
    finally:
        logger.info("Application shutdown: cancelling background initialization tasks...")
        await detect.GLOBAL_MODEL_A3_ENGINE.cancel_background_init()


app = FastAPI(
    title="RoomAI — 3D Spatial Understanding & Scene Reconstruction API",
    version="1.4.0",
    description=(
        "Production REST API for indoor scene perception, metric monocular & multi-view depth estimation, "
        "dense 3D point cloud generation, floor layout recovery, and validated glTF 2.0 / PLY asset serialization."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration: allows strict origin restriction in staging/production via ROOMAI_CORS_ORIGINS
_cors_env = os.getenv("ROOMAI_CORS_ORIGINS", "").strip()
if _cors_env:
    allowed_origins = [orig.strip() for orig in _cors_env.split(",") if orig.strip()]
    allow_origin_regex = None
    allow_credentials = False if "*" in allowed_origins else True
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    allow_origin_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_and_dos_protection(request: Request, call_next):
    """Logs request lifecycle and attaches X-Process-Time-Ms header."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardized JSON envelope for HTTP exceptions."""
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", f"HTTP_{exc.status_code}")
        message = detail.get("message", str(detail))
        remediation = detail.get("remediation")
    else:
        code = f"HTTP_{exc.status_code}"
        message = str(detail)
        remediation = None

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            error=ErrorDetail(code=code, message=message, remediation=remediation),
            timestamp=time.time(),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all unhandled exception handler."""
    logger.critical(f"Unhandled server exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            status="error",
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message=str(exc),
                remediation="Check server logs or contact RoomAI engineering.",
            ),
            timestamp=time.time(),
        ).model_dump(),
    )


# Register Routers
app.include_router(health.router)
app.include_router(uploads.router)
app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(detect.router)
app.include_router(experimental.router)


@app.get("/", tags=["Root"])
def root():
    """API Welcome and Service Discovery."""
    return {
        "service": "RoomAI 3D Scene Reconstruction API",
        "version": "1.4.0",
        "documentation": "/docs",
        "openapi_schema": "/api/v1/openapi.json",
        "health": "/health",
        "readiness": "/ready",
        "api_v1": {
            "detect": "/api/v1/detect",
            "uploads": "/api/v1/uploads",
            "jobs": "/api/v1/jobs",
            "models": "/api/v1/models",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)

