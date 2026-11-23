from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_name: str
    device: str
    version: str


class TriageAssessment(BaseModel):
    referable_dr: bool = Field(
        ...,
        description="True if predicted severity is Moderate, Severe, or Proliferative DR.",
    )
    triage_status: Literal["ROUTINE_SCREENING", "NEEDS_SPECIALIST_REVIEW"]
    review_reasons: list[str] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    predicted_class: str
    predicted_class_index: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    entropy: float = Field(..., ge=0.0)
    calibration_temperature: float
    probabilities: dict[str, float]
    triage: TriageAssessment
    inference_time_ms: float


class ExplanationResponse(PredictionResponse):
    cam_method: str
    target_layer: str
    gradcam_overlay_base64: str = Field(
        ...,
        description="Base64-encoded PNG image showing Grad-CAM overlay on the retinal image.",
    )


class PredictionHistoryItem(BaseModel):
    request_id: str
    endpoint: str
    model_name: str
    predicted_class: str
    confidence: float
    entropy: float
    referable_dr: bool
    triage_status: str
    inference_time_ms: float
    created_at: str
