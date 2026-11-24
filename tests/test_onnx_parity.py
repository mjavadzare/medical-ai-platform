"""Automated ONNX export and PyTorch vs. ONNX Runtime numerical parity tests."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet50

from modules.diabetic_retinopathy.export_onnx import (
    create_ort_session,
    export_model_to_onnx,
    verify_parity,
)
from modules.diabetic_retinopathy.inference.predictor import DRPredictor


def test_onnx_export_and_predictor_parity(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.eval()

    checkpoint_path = tmp_path / "resnet50_test.pth"
    onnx_path = tmp_path / "resnet50_test.onnx"
    torch.save(model.state_dict(), checkpoint_path)

    export_model_to_onnx(
        model=model, onnx_path=onnx_path, image_size=224, opset_version=17
    )
    assert onnx_path.exists()

    session = create_ort_session(onnx_path)
    parity = verify_parity(
        model=model,
        session=session,
        temperature=0.9196,
        image_size=224,
        batch_sizes=(1, 2),
        atol=1e-4,
    )
    assert parity["passed"] is True
    assert parity["argmax_match"] is True

    rng = np.random.default_rng(7)
    pixels = rng.integers(20, 235, size=(256, 256, 3), dtype=np.uint8)
    test_image = Image.fromarray(pixels, mode="RGB")

    pt_predictor = DRPredictor(
        checkpoint_path=checkpoint_path,
        calibration_temperature=0.9196,
        onnx_path=onnx_path,
        use_onnx=False,
    )
    onnx_predictor = DRPredictor(
        checkpoint_path=checkpoint_path,
        calibration_temperature=0.9196,
        onnx_path=onnx_path,
        use_onnx=True,
    )

    pt_predictor.load()
    onnx_predictor.load()
    try:
        assert pt_predictor.inference_backend == "pytorch"
        assert onnx_predictor.inference_backend == "onnxruntime"

        pt_result = pt_predictor.predict(test_image)
        onnx_result = onnx_predictor.predict(test_image)

        assert pt_result["predicted_class"] == onnx_result["predicted_class"]
        assert (
            pt_result["triage"]["triage_status"]
            == onnx_result["triage"]["triage_status"]
        )
        for cls_name, pt_prob in pt_result["probabilities"].items():
            assert abs(pt_prob - onnx_result["probabilities"][cls_name]) < 1e-4
    finally:
        pt_predictor.unload()
        onnx_predictor.unload()
