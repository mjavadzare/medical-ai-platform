from PIL import Image
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import PredictionLog
from app.schemas.dr import ExplanationResponse, PredictionResponse
from modules.diabetic_retinopathy.inference.predictor import DRPredictor


class DiabeticRetinopathyService:
    """Application service orchestrating DRPredictor and audit logging."""

    def __init__(self, settings: Settings):
        self.predictor = DRPredictor(
            checkpoint_path=settings.checkpoint_path,
            model_name=settings.model_name,
            image_size=settings.image_size,
            cam_method=settings.cam_method,
            cam_layer=settings.cam_layer,
            calibration_temperature=settings.calibration_temperature,
            triage_confidence_threshold=settings.triage_confidence_threshold,
            triage_entropy_threshold=settings.triage_entropy_threshold,
        )

    def load(self) -> None:
        self.predictor.load()

    def unload(self) -> None:
        self.predictor.unload()

    @property
    def is_ready(self) -> bool:
        return self.predictor.is_ready

    @property
    def device(self):
        return self.predictor.device

    @staticmethod
    def _log_prediction(
        db: Session | None,
        endpoint: str,
        payload: dict,
    ) -> None:
        if db is None:
            return
        record = PredictionLog(
            request_id=payload["request_id"],
            endpoint=endpoint,
            model_name=payload["model_name"],
            predicted_class=payload["predicted_class"],
            confidence=payload["confidence"],
            entropy=payload["entropy"],
            referable_dr=payload["triage"]["referable_dr"],
            triage_status=payload["triage"]["triage_status"],
            inference_time_ms=payload["inference_time_ms"],
        )
        db.add(record)
        db.commit()

    def predict(
        self,
        image: Image.Image,
        db: Session | None = None,
    ) -> PredictionResponse:
        result = self.predictor.predict(image)
        self._log_prediction(db=db, endpoint="predict", payload=result)
        return PredictionResponse(**result)

    def explain(
        self,
        image: Image.Image,
        db: Session | None = None,
    ) -> ExplanationResponse:
        result = self.predictor.explain(image)
        self._log_prediction(db=db, endpoint="explain", payload=result)
        return ExplanationResponse(**result)

    @staticmethod
    def get_history(db: Session, limit: int = 20) -> list[PredictionLog]:
        stmt = (
            select(PredictionLog)
            .order_by(desc(PredictionLog.created_at))
            .limit(limit)
        )
        return list(db.scalars(stmt).all())
