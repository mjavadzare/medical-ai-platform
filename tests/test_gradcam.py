import pytest
import torch
from torch import nn

from shared.explainability.gradcam import GradCAM, GradCAMPlusPlus


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.mark.parametrize("cam_cls", [GradCAM, GradCAMPlusPlus])
def test_cam_output_shape_range_and_hook_cleanup(cam_cls):
    torch.manual_seed(0)
    model = TinyCNN(num_classes=5).eval()
    input_tensor = torch.randn(1, 3, 32, 32)

    with cam_cls(model=model, target_layer=model.conv2) as cam_engine:
        cam, pred_class, probs = cam_engine.generate(input_tensor=input_tensor)

        assert cam.ndim == 2
        assert cam.shape == (16, 16)
        assert float(cam.min()) >= 0.0
        assert float(cam.max()) <= 1.0 + 1e-5
        assert 0 <= pred_class < 5
        assert probs.shape == (5,)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)

    assert cam_engine.forward_handle is None
    assert cam_engine.backward_handle is None
