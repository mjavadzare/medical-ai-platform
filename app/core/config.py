import os
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    app_name: str = "Medical AI Platform — Diabetic Retinopathy API"
    app_version: str = "1.1.0"
    api_v1_prefix: str = "/api/v1"

    model_name: str = os.getenv(
        "MODEL_NAME",
        "ResNet50_layer3_layer4_ft_bn_frozen_focal",
    )
    checkpoint_path: Path = Path(
        os.getenv(
            "CHECKPOINT_PATH",
            str(
                PROJECT_ROOT
                / "modules"
                / "diabetic_retinopathy"
                / "artifacts"
                / "checkpoints"
                / "ResNet50_layer3_layer4_ft_bn_frozen_focal"
                / "ResNet50_layer3_layer4_ft_bn_frozen_focal_best_model.pth"
            ),
        )
    )
    onnx_path: Path = Path(
        os.getenv(
            "ONNX_PATH",
            str(
                PROJECT_ROOT
                / "modules"
                / "diabetic_retinopathy"
                / "artifacts"
                / "onnx"
                / "ResNet50_layer3_layer4_ft_bn_frozen_focal.onnx"
            ),
        )
    )
    use_onnx: bool = os.getenv("USE_ONNX", "true").lower() in {"1", "true", "yes"}

    num_classes: int = 5
    class_names: list[str] = Field(
        default_factory=lambda: [
            "No DR",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative DR",
        ]
    )
    referable_classes: set[str] = Field(
        default_factory=lambda: {
            "Moderate",
            "Severe",
            "Proliferative DR",
        }
    )

    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    cam_method: str = os.getenv("CAM_METHOD", "gradcam++")
    cam_layer: str = os.getenv("CAM_LAYER", "layer4")

    # Post-hoc calibration & clinical triage thresholds
    calibration_temperature: float = float(
        os.getenv("CALIBRATION_TEMPERATURE", "0.9196")
    )
    triage_confidence_threshold: float = float(
        os.getenv("TRIAGE_CONFIDENCE_THRESHOLD", "0.75")
    )
    triage_entropy_threshold: float = float(
        os.getenv("TRIAGE_ENTROPY_THRESHOLD", "1.05")
    )

    # Audit log database (SQLite by default; override with PostgreSQL URL in Docker)
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(PROJECT_ROOT / 'medical_ai.db').as_posix()}",
    )


settings = Settings()
