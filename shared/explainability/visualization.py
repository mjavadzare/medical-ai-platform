from pathlib import Path

import numpy as np

from PIL import Image


def normalize_cam(cam):
    """
    Normalize Grad-CAM values to the range [0, 1].
    """

    cam = np.asarray(cam, dtype=np.float32)

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam)

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min
    )

    return cam


def create_heatmap(
    cam,
    image_size
):
    """
    Convert a Grad-CAM array into a heatmap image.
    """

    cam = normalize_cam(cam)

    # Convert [0, 1] -> [0, 255]
    cam_uint8 = (
        cam * 255
    ).astype(np.uint8)

    heatmap = Image.fromarray(
        cam_uint8,
        mode="L"
    )

    heatmap = heatmap.resize(
        image_size,
        Image.Resampling.BILINEAR
    )

    return heatmap


def create_gradcam_overlay(
    image,
    cam,
    alpha=0.45
):
    """
    Create a colored Grad-CAM overlay
    on top of the original image.
    """

    image = image.convert("RGB")

    heatmap = create_heatmap(
        cam=cam,
        image_size=image.size
    )

    # Convert grayscale CAM to NumPy
    heatmap_array = np.asarray(
        heatmap,
        dtype=np.float32
    ) / 255.0

    # Create a simple JET-like heatmap manually.
    # This avoids adding another dependency.
    red = np.clip(
        1.5 * heatmap_array - 0.5,
        0.0,
        1.0
    )

    green = np.clip(
        1.5
        - np.abs(
            4.0 * heatmap_array - 2.0
        ),
        0.0,
        1.0
    )

    blue = np.clip(
        1.5 * (1.0 - heatmap_array) - 0.5,
        0.0,
        1.0
    )

    heatmap_rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    heatmap_rgb = (
        heatmap_rgb * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        heatmap_rgb,
        mode="RGB"
    )

    overlay = Image.blend(
        image,
        heatmap_image,
        alpha=alpha
    )

    return (
        heatmap_image,
        overlay
    )


def save_gradcam_outputs(
    image,
    cam,
    output_dir,
    image_stem
):
    """
    Save original image, heatmap and overlay.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    heatmap, overlay = (
        create_gradcam_overlay(
            image=image,
            cam=cam
        )
    )

    original_path = (
        output_dir
        / f"{image_stem}_original.png"
    )

    heatmap_path = (
        output_dir
        / f"{image_stem}_heatmap.png"
    )

    overlay_path = (
        output_dir
        / f"{image_stem}_overlay.png"
    )

    image.save(
        original_path
    )

    heatmap.save(
        heatmap_path
    )

    overlay.save(
        overlay_path
    )

    return (
        original_path,
        heatmap_path,
        overlay_path
    )