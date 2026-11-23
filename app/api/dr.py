from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.dr import (
    ExplanationResponse,
    PredictionHistoryItem,
    PredictionResponse,
)
from app.services.dr_service import DiabeticRetinopathyService
from app.utils.image_validation import validate_and_load_fundus_image

router = APIRouter(prefix="/dr", tags=["Diabetic Retinopathy"])


def get_dr_service(request: Request) -> DiabeticRetinopathyService:
    service: DiabeticRetinopathyService | None = getattr(
        request.app.state, "dr_service", None
    )
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Diabetic Retinopathy model is not loaded.",
        )
    return service


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify retinal fundus image",
)
async def predict_dr(
    file: Annotated[
        UploadFile,
        File(description="Retinal fundus image (JPEG or PNG)"),
    ],
    service: Annotated[DiabeticRetinopathyService, Depends(get_dr_service)],
    db: Annotated[Session, Depends(get_db)],
):
    image = await validate_and_load_fundus_image(file)
    return service.predict(image=image, db=db)


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Classify retinal fundus image and generate Grad-CAM++ overlay",
)
async def explain_dr(
    file: Annotated[
        UploadFile,
        File(description="Retinal fundus image (JPEG or PNG)"),
    ],
    service: Annotated[DiabeticRetinopathyService, Depends(get_dr_service)],
    db: Annotated[Session, Depends(get_db)],
):
    image = await validate_and_load_fundus_image(file)
    return service.explain(image=image, db=db)


@router.get(
    "/history",
    response_model=list[PredictionHistoryItem],
    summary="Retrieve recent prediction audit logs",
)
def get_prediction_history(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    records = DiabeticRetinopathyService.get_history(db=db, limit=limit)
    return [
        PredictionHistoryItem(
            request_id=r.request_id,
            endpoint=r.endpoint,
            model_name=r.model_name,
            predicted_class=r.predicted_class,
            confidence=r.confidence,
            entropy=r.entropy,
            referable_dr=r.referable_dr,
            triage_status=r.triage_status,
            inference_time_ms=r.inference_time_ms,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]
