import base64
import io

import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

REFERABLE_CLASSES = {
    "Moderate",
    "Severe",
    "Proliferative DR",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def create_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Standard deterministic transform matching ResNet50 evaluation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def render_cam_overlay_base64(image: Image.Image, cam: torch.Tensor) -> str:
    """Blend a normalized [0, 1] CAM tensor over the RGB image and return Base64 PNG."""
    rgb_image = image.convert("RGB")
    cam_np = (cam.detach().cpu().numpy() * 255).astype(np.uint8)
    cam_resized = Image.fromarray(cam_np).resize(
        rgb_image.size, resample=Image.Resampling.BILINEAR
    )
    cam_norm = np.asarray(cam_resized, dtype=np.float32) / 255.0

    colormap = matplotlib.colormaps["jet"]
    heatmap_rgb = (colormap(cam_norm)[:, :, :3] * 255).astype(np.float32)
    orig_rgb = np.asarray(rgb_image, dtype=np.float32)

    blended = np.clip(0.55 * orig_rgb + 0.45 * heatmap_rgb, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(blended)

    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
