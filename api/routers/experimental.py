"""Experimental and Offline Feature Endpoints (Truth-in-Advertising)."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/experimental", tags=["Experimental & Offline Features"])


class VQARequest(BaseModel):
    upload_id: str
    question: str


class RedesignRequest(BaseModel):
    upload_id: str
    style_prompt: str


@router.post("/vqa")
def ask_vqa(req: VQARequest):
    """Vision Question Answering endpoint (Currently Offline)."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "feature": "Vision Question Answering",
            "status": "UNAVAILABLE",
            "reason": "Multimodal vision-language model weights (e.g. Qwen2-VL, MiniCPM-V) are not installed locally.",
            "remediation": "This feature is disabled in production until weights and GPU quota are allocated.",
        },
    )


@router.post("/redesign")
def generate_redesign(req: RedesignRequest):
    """Generative Interior Redesign endpoint (Currently Offline)."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "feature": "Generative Room Redesign",
            "status": "UNAVAILABLE",
            "reason": "Diffusion pipeline checkpoints (SDXL / ControlNet Depth) are not provisioned in local storage.",
            "remediation": "This capability is disabled to prevent misleading mock generation.",
        },
    )

