import io
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dr import get_dr_service
from app.db.base import Base
from app.db.models import PredictionLog
from app.db.session import get_db
from app.main import app
from app.schemas.dr import ExplanationResponse, PredictionResponse, TriageAssessment


class DummyDRService:
    is_ready = True
    device = "cpu"

    def _build_prediction(self) -> PredictionResponse:
        return PredictionResponse(
            request_id=str(uuid.uuid4()),
            model_name="dummy-resnet50",
            predicted_class="Moderate",
            predicted_class_index=2,
            confidence=0.82,
            entropy=0.61,
            calibration_temperature=1.0,
            probabilities={
                "No DR": 0.02,
                "Mild": 0.12,
                "Moderate": 0.82,
                "Severe": 0.03,
                "Proliferative DR": 0.01,
            },
            triage=TriageAssessment(
                referable_dr=True,
                triage_status="NEEDS_SPECIALIST_REVIEW",
                review_reasons=["Referable DR detected (Moderate)."],
            ),
            inference_time_ms=12.4,
        )

    @staticmethod
    def _record(db: Session | None, endpoint: str, resp: PredictionResponse) -> None:
        if db is None:
            return
        db.add(
            PredictionLog(
                request_id=resp.request_id,
                endpoint=endpoint,
                model_name=resp.model_name,
                predicted_class=resp.predicted_class,
                confidence=resp.confidence,
                entropy=resp.entropy,
                referable_dr=resp.triage.referable_dr,
                triage_status=resp.triage.triage_status,
                inference_time_ms=resp.inference_time_ms,
            )
        )
        db.commit()

    def predict(
        self,
        image: Image.Image,
        db: Session | None = None,
    ) -> PredictionResponse:
        resp = self._build_prediction()
        self._record(db, "predict", resp)
        return resp

    def explain(
        self,
        image: Image.Image,
        db: Session | None = None,
    ) -> ExplanationResponse:
        pred = self._build_prediction()
        self._record(db, "explain", pred)
        return ExplanationResponse(
            **pred.model_dump(),
            cam_method="gradcam++",
            target_layer="layer4",
            gradcam_overlay_base64=(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                "+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
        )


def _create_sample_png_bytes(size: int = 128, solid: bool = False) -> bytes:
    if solid:
        arr = np.full((size, size, 3), 120, dtype=np.uint8)
    else:
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    dummy_service = DummyDRService()
    app.dependency_overrides[get_dr_service] = lambda: dummy_service
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        app.state.dr_service = dummy_service
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_and_history_endpoints(client: TestClient):
    response = client.post(
        "/api/v1/dr/predict",
        files={"file": ("fundus.png", _create_sample_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "Moderate"
    assert data["triage"]["referable_dr"] is True
    assert data["triage"]["triage_status"] == "NEEDS_SPECIALIST_REVIEW"

    history_response = client.get("/api/v1/dr/history?limit=10")
    assert history_response.status_code == 200
    records = history_response.json()
    assert len(records) == 1
    assert records[0]["request_id"] == data["request_id"]
    assert records[0]["endpoint"] == "predict"


def test_explain_endpoint(client: TestClient):
    response = client.post(
        "/api/v1/dr/explain",
        files={"file": ("fundus.png", _create_sample_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["cam_method"] == "gradcam++"
    assert data["target_layer"] == "layer4"
    assert len(data["gradcam_overlay_base64"]) > 20


def test_reject_unsupported_file_type(client: TestClient):
    response = client.post(
        "/api/v1/dr/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_reject_low_variance_blank_image(client: TestClient):
    response = client.post(
        "/api/v1/dr/predict",
        files={"file": ("blank.png", _create_sample_png_bytes(solid=True), "image/png")},
    )
    assert response.status_code == 422


def test_reject_too_small_image(client: TestClient):
    response = client.post(
        "/api/v1/dr/predict",
        files={"file": ("tiny.png", _create_sample_png_bytes(size=32), "image/png")},
    )
    assert response.status_code == 422
