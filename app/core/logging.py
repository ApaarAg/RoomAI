"""RoomAI Backend Structured Logging.

JSON-formatted logs with trace ID propagation for observability.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional


# Context variables for trace correlation
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
job_id_var: ContextVar[Optional[str]] = ContextVar("job_id", default=None)
model_id_var: ContextVar[Optional[str]] = ContextVar("model_id", default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter with trace context."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add trace context if available
        trace_id = trace_id_var.get()
        if trace_id:
            log_obj["trace_id"] = trace_id

        job_id = job_id_var.get()
        if job_id:
            log_obj["job_id"] = job_id

        model_id = model_id_var.get()
        if model_id:
            log_obj["model_id"] = model_id

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "name", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info"
            }:
                log_obj[key] = value

        # Include exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def get_logger(name: str = "RoomAI") -> logging.Logger:
    """Get or create a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class TraceContext:
    """Context manager for trace ID propagation."""

    def __init__(
        self,
        trace_id: Optional[str] = None,
        job_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.job_id = job_id
        self.model_id = model_id
        self.tokens = []

    def __enter__(self) -> "TraceContext":
        self.tokens.append(trace_id_var.set(self.trace_id))
        if self.job_id:
            self.tokens.append(job_id_var.set(self.job_id))
        if self.model_id:
            self.tokens.append(model_id_var.set(self.model_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in reversed(self.tokens):
            try:
                if token.var is trace_id_var:
                    trace_id_var.reset(token)
                elif token.var is job_id_var:
                    job_id_var.reset(token)
                elif token.var is model_id_var:
                    model_id_var.reset(token)
            except ValueError:
                pass


def log_job_event(
    logger: logging.Logger,
    event: str,
    job_id: str,
    status: str,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    duration_ms: Optional[float] = None,
    **extra: Any,
) -> None:
    """Log a structured job lifecycle event."""
    with TraceContext(job_id=job_id):
        logger.info(
            event,
            extra={
                "event_type": "job_lifecycle",
                "job_id": job_id,
                "status": status,
                "stage": stage,
                "progress": progress,
                "duration_ms": duration_ms,
                **extra,
            },
        )


def log_inference_event(
    logger: logging.Logger,
    event: str,
    model_id: str,
    input_shape: Optional[tuple] = None,
    output_shape: Optional[tuple] = None,
    duration_ms: Optional[float] = None,
    vram_mb: Optional[float] = None,
    **extra: Any,
) -> None:
    """Log a structured inference event."""
    with TraceContext(model_id=model_id):
        logger.info(
            event,
            extra={
                "event_type": "inference",
                "model_id": model_id,
                "input_shape": str(input_shape) if input_shape else None,
                "output_shape": str(output_shape) if output_shape else None,
                "duration_ms": duration_ms,
                "vram_mb": vram_mb,
                **extra,
            },
        )


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    trace_id: str,
    status_code: int,
    duration_ms: float,
    **extra: Any,
) -> None:
    """Log a structured API request."""
    with TraceContext(trace_id=trace_id):
        logger.info(
            f"{method} {path}",
            extra={
                "event_type": "api_request",
                "method": method,
                "path": path,
                "trace_id": trace_id,
                "status_code": status_code,
                "duration_ms": duration_ms,
                **extra,
            },
        )