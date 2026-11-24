import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet50

from modules.diabetic_retinopathy.inference.preprocessing import (
    CLASS_NAMES,
    REFERABLE_CLASSES,
    create_inference_transform,
    render_cam_overlay_base64,
)
from shared.explainability.gradcam import GradCAM, GradCAMPlusPlus

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class DRPredictor:
    """
    Standalone inference & Grad-CAM++ engine for Diabetic Retinopathy.
    Decoupled from FastAPI so it can be reused in API, CLI, or batch jobs.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str = "ResNet50_layer3_layer4_ft_bn_frozen_focal",
        image_size: int = 224,
        cam_method: str = "gradcam++",
        cam_layer: str = "layer4",
        calibration_temperature: float = 1.0,
        triage_confidence_threshold: float = 0.75,
        triage_entropy_threshold: float = 1.05,
        onnx_path: Path | None = None,
        use_onnx: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.image_size = image_size
        self.cam_method = cam_method
        self.cam_layer = cam_layer
        self.calibration_temperature = calibration_temperature
        self.triage_confidence_threshold = triage_confidence_threshold
        self.triage_entropy_threshold = triage_entropy_threshold
        self.onnx_path = Path(onnx_path) if onnx_path is not None else None
        self.use_onnx = use_onnx

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = create_inference_transform(image_size=image_size)
        self.model: torch.nn.Module | None = None
        self.cam_engine: GradCAM | GradCAMPlusPlus | None = None
        self.ort_session: Any | None = None

    def load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

        target_layer = (
            model.layer3[-1].conv3
            if self.cam_layer == "layer3"
            else model.layer4[-1].conv3
        )
        cam_cls = GradCAMPlusPlus if self.cam_method == "gradcam++" else GradCAM
        self.cam_engine = cam_cls(model=model, target_layer=target_layer)

        self.ort_session = None
        if (
            self.use_onnx
            and ort is not None
            and self.onnx_path is not None
            and self.onnx_path.exists()
        ):
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

    def unload(self) -> None:
        if self.cam_engine is not None:
            self.cam_engine.remove_hooks()
            self.cam_engine = None
        self.ort_session = None
        self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.cam_engine is not None

    @property
    def inference_backend(self) -> str:
        return "onnxruntime" if self.ort_session is not None else "pytorch"

    @staticmethod
    def _compute_entropy(probs: list[float]) -> float:
        eps = 1e-12
        return -sum(p * math.log(max(p, eps)) for p in probs if p > 0)

    def _build_triage(
        self,
        predicted_class: str,
        confidence: float,
        entropy: float,
    ) -> dict:
        referable = predicted_class in REFERABLE_CLASSES
        reasons: list[str] = []

        if referable:
            reasons.append(f"Referable DR detected ({predicted_class}).")
        if confidence < self.triage_confidence_threshold:
            reasons.append(
                f"Calibrated confidence ({confidence:.2f}) below threshold "
                f"({self.triage_confidence_threshold:.2f})."
            )
        if entropy > self.triage_entropy_threshold:
            reasons.append(f"High predictive uncertainty (entropy={entropy:.2f}).")

        return {
            "referable_dr": referable,
            "triage_status": (
                "NEEDS_SPECIALIST_REVIEW" if reasons else "ROUTINE_SCREENING"
            ),
            "review_reasons": reasons,
        }

    def _infer_calibrated_probs(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[int, list[float]]:
        temp = max(self.calibration_temperature, 1e-6)

        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            np_input = input_tensor.detach().cpu().numpy().astype(np.float32)
            logits = self.ort_session.run(None, {input_name: np_input})[0][0]
            scaled = logits / temp
            shifted = scaled - float(np.max(scaled))
            exps = np.exp(shifted)
            probs_np = exps / float(np.sum(exps))
            pred_idx = int(np.argmax(probs_np))
            return pred_idx, probs_np.astype(float).tolist()

        if self.model is None:
            raise RuntimeError("DRPredictor PyTorch model is not loaded.")

        with torch.no_grad():
            logits = self.model(input_tensor)
            scaled = logits / temp
            probs = torch.softmax(scaled, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        return pred_idx, probs.cpu().tolist()

    def predict(self, image: Image.Image) -> dict:
        if not self.is_ready:
            raise RuntimeError("DRPredictor is not loaded.")

        start = time.perf_counter()
        rgb_image = image.convert("RGB")
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        pred_idx, probs_list = self._infer_calibrated_probs(input_tensor)

        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs_list[pred_idx])
        entropy = self._compute_entropy(probs_list)

        return {
            "request_id": str(uuid.uuid4()),
            "model_name": self.model_name,
            "predicted_class": pred_class,
            "predicted_class_index": pred_idx,
            "confidence": round(confidence, 6),
            "entropy": round(entropy, 6),
            "calibration_temperature": self.calibration_temperature,
            "probabilities": {
                name: round(float(probs_list[i]), 6)
                for i, name in enumerate(CLASS_NAMES)
            },
            "triage": self._build_triage(pred_class, confidence, entropy),
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    def explain(self, image: Image.Image) -> dict:
        if not self.is_ready or self.cam_engine is None:
            raise RuntimeError("DRPredictor explainability engine is not loaded.")

        start = time.perf_counter()
        rgb_image = image.convert("RGB")
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        cam, pred_idx, raw_probs = self.cam_engine.generate(input_tensor=input_tensor)

        if abs(self.calibration_temperature - 1.0) > 1e-6:
            pred_idx, probs_list = self._infer_calibrated_probs(input_tensor)
        else:
            probs_list = raw_probs.cpu().tolist()

        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs_list[pred_idx])
        entropy = self._compute_entropy(probs_list)
        overlay_b64 = render_cam_overlay_base64(rgb_image, cam)

        return {
            "request_id": str(uuid.uuid4()),
            "model_name": self.model_name,
            "predicted_class": pred_class,
            "predicted_class_index": pred_idx,
            "confidence": round(confidence, 6),
            "entropy": round(entropy, 6),
            "calibration_temperature": self.calibration_temperature,
            "probabilities": {
                name: round(float(probs_list[i]), 6)
                for i, name in enumerate(CLASS_NAMES)
            },
            "triage": self._build_triage(pred_class, confidence, entropy),
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "cam_method": self.cam_method,
            "target_layer": self.cam_layer,
            "gradcam_overlay_base64": overlay_b64,
        }
