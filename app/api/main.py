"""RoomAI Backend API - FastAPI Application."""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import RoomAIException
from app.core.logging import get_logger, log_api_request, TraceContext
from app.monitoring.metrics import get_metrics_collector
from app.workers.gpu_worker import get_gpu_worker, shutdown_gpu_worker

from app.api.routers import health, models, uploads, jobs, exports


logger = get_logger("RoomAI_API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RoomAI Backend API")
    get_gpu_worker()  # Initialize GPU worker
    logger.info("GPU Worker started")

    yield

    # Shutdown
    logger.info("Shutting down RoomAI Backend API")
    shutdown_gpu_worker()
    logger.info("GPU Worker stopped")


app = FastAPI(
    title="RoomAI — 3D Spatial Understanding & Scene Reconstruction API",
    version=settings.app_version,
    description=(
        "Production REST API for indoor scene perception, metric monocular & multi-view depth estimation, "
        "dense 3D point cloud generation, floor layout recovery, and validated glTF 2.0 / PLY asset serialization."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_and_metrics(request: Request, call_next):
    """Add request timing and collect metrics."""
    trace_id = request.headers.get("x-trace-id", f"tr_{int(time.time() * 1000)}")
    t0 = time.perf_counter()

    with TraceContext(trace_id=trace_id):
        response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    response.headers["X-Trace-Id"] = trace_id

    # Record metrics
    metrics = get_metrics_collector()
    metrics.record_api_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=elapsed_ms,
    )

    log_api_request(
        logger,
        method=request.method,
        path=request.url.path,
        trace_id=trace_id,
        status_code=response.status_code,
        duration_ms=elapsed_ms,
    )

    return response


@app.exception_handler(RoomAIException)
async def roomai_exception_handler(request: Request, exc: RoomAIException):
    """Handle RoomAI structured exceptions."""
    from app.core.exceptions import exception_to_response
    return JSONResponse(
        status_code=exc.status_code,
        content=exception_to_response(exc),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTP exceptions."""
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
        content={
            "status": "error",
            "error": {
                "code": code,
                "message": message,
                "remediation": remediation,
            },
            "timestamp": time.time(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all unhandled exception handler."""
    logger.critical(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": str(exc),
                "remediation": "Check server logs or contact RoomAI engineering.",
            },
            "timestamp": time.time(),
        },
    )


# Register routers
app.include_router(health.router)
app.include_router(uploads.router)
app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(exports.router)


@app.get("/", tags=["Root"])
def root():
    """API Welcome and Service Discovery."""
    return {
        "service": "RoomAI 3D Scene Reconstruction API",
        "version": settings.app_version,
        "documentation": "/docs",
        "openapi_schema": "/api/v1/openapi.json",
        "health": "/health",
        "readiness": "/ready",
        "metrics": "/metrics",
        "api_v1": {
            "uploads": "/api/v1/uploads",
            "jobs": "/api/v1/jobs",
            "models": "/api/v1/models",
            "exports": "/api/v1/exports",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )