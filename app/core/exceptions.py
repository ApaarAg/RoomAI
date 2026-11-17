"""RoomAI Backend Exceptions.

Structured exception hierarchy for clean error handling and API responses.
"""

from typing import Any, Dict, Optional


class RoomAIException(Exception):
    """Base exception for all RoomAI errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        remediation: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        self.remediation = remediation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
        }


class ValidationError(RoomAIException):
    """Input validation failed."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, code="VALIDATION_ERROR", status_code=400, **kwargs)
        if field:
            self.details["field"] = field


class NotFoundError(RoomAIException):
    """Resource not found."""
    def __init__(self, resource: str, identifier: str, **kwargs):
        super().__init__(
            f"{resource} '{identifier}' not found",
            code="NOT_FOUND",
            status_code=404,
            details={"resource": resource, "identifier": identifier},
            **kwargs
        )


class UnavailableError(RoomAIException):
    """Service or model temporarily unavailable."""
    def __init__(self, message: str, reason: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="SERVICE_UNAVAILABLE",
            status_code=503,
            details={"reason": reason} if reason else {},
            **kwargs
        )


class GPUError(RoomAIException):
    """GPU-related error (OOM, driver, etc.)."""
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="GPU_ERROR",
            status_code=500,
            details={"operation": operation} if operation else {},
            remediation="Reduce batch size, resolution, or view count; retry with smaller model",
            **kwargs
        )


class ModelError(RoomAIException):
    """Model loading or inference error."""
    def __init__(self, message: str, model_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="MODEL_ERROR",
            status_code=500,
            details={"model_id": model_id} if model_id else {},
            **kwargs
        )


class InferenceError(RoomAIException):
    """Model inference error."""
    def __init__(self, message: str, model_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="INFERENCE_ERROR",
            status_code=500,
            details={"model_id": model_id} if model_id else {},
            **kwargs
        )


class JobError(RoomAIException):
    """Job orchestration error."""
    def __init__(self, message: str, job_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="JOB_ERROR",
            status_code=500,
            details={"job_id": job_id} if job_id else {},
            **kwargs
        )


class CancellationError(RoomAIException):
    """Job was cancelled by user or system."""
    def __init__(self, job_id: str, **kwargs):
        super().__init__(
            f"Job {job_id} was cancelled",
            code="JOB_CANCELLED",
            status_code=499,
            details={"job_id": job_id},
            **kwargs
        )


class ExportError(RoomAIException):
    """3D asset export validation failed."""
    def __init__(self, message: str, format: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="EXPORT_ERROR",
            status_code=500,
            details={"format": format} if format else {},
            **kwargs
        )


class ReconstructionError(RoomAIException):
    """Reconstruction pipeline error."""
    def __init__(self, message: str, job_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="RECONSTRUCTION_ERROR",
            status_code=500,
            details={"job_id": job_id} if job_id else {},
            **kwargs
        )


class PreprocessingError(RoomAIException):
    """Image preprocessing failed."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            code="PREPROCESSING_ERROR",
            status_code=400,
            **kwargs
        )


class ConfigurationError(RoomAIException):
    """Configuration or setup error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            code="CONFIGURATION_ERROR",
            status_code=500,
            **kwargs
        )


# HTTP status code to exception mapping
STATUS_TO_EXCEPTION = {
    400: ValidationError,
    404: NotFoundError,
    499: CancellationError,
    500: RoomAIException,
    503: UnavailableError,
}


def exception_to_response(exc: RoomAIException) -> Dict[str, Any]:
    """Convert exception to standardized API error response."""
    return {
        "status": "error",
        "error": exc.to_dict(),
        "timestamp": __import__("time").time(),
    }