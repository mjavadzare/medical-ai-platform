from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    endpoint: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(128))
    predicted_class: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    entropy: Mapped[float] = mapped_column(Float)
    referable_dr: Mapped[bool] = mapped_column(Boolean)
    triage_status: Mapped[str] = mapped_column(String(64))
    inference_time_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
