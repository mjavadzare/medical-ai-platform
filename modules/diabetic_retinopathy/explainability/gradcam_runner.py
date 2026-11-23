import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from shared.explainability.gradcam import GradCAM, GradCAMPlusPlus
from shared.explainability.visualization import save_gradcam_outputs


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_MODEL_NAME = "ResNet50_layer3_layer4_ft_bn_frozen_focal"
NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

CLASS_NAME_MAPPING = {
    "No_DR": "No DR",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "Proliferate_DR": "Proliferative DR",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples" / "diabetic_retinopathy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def create_model(checkpoint_path: Path) -> torch.nn.Module:
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=NUM_CLASSES,
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------

def create_transform(resize: int | None = None):
    weights_meta = ResNet50_Weights.DEFAULT.transforms()
    steps = []

    if resize is not None:
        steps.append(transforms.Resize((resize, resize)))

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights_meta.mean,
            std=weights_meta.std,
        ),
    ])

    return transforms.Compose(steps)


# ---------------------------------------------------------
# CAM Engine
# ---------------------------------------------------------

def resolve_target_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    if layer_name == "layer4":
        return model.layer4[-1].conv3
    if layer_name == "layer3":
        return model.layer3[-1].conv3
    raise ValueError(
        f"Unsupported layer '{layer_name}'. Choose 'layer4' or 'layer3'.")


def create_cam_engine(model: torch.nn.Module, method: str, layer_name: str):
    target_layer = resolve_target_layer(model, layer_name)

    if method == "gradcam":
        return GradCAM(model=model, target_layer=target_layer)
    if method == "gradcam++":
        return GradCAMPlusPlus(model=model, target_layer=target_layer)

    raise ValueError(
        f"Unsupported method '{method}'. Choose 'gradcam' or 'gradcam++'.")


# ---------------------------------------------------------
# Image & Label Helpers
# ---------------------------------------------------------

def load_image(image_path: Path, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image, input_tensor


def normalize_class_name(class_name: str) -> str:
    return CLASS_NAME_MAPPING.get(class_name, class_name)


def find_images(samples_dir: Path) -> list[Path]:
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []

    for class_dir in samples_dir.iterdir():
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                image_paths.append(image_path)

    return sorted(image_paths)


# ---------------------------------------------------------
# Generate Explanation
# ---------------------------------------------------------

def explain_image(
    image_path: Path,
    transform,
    cam_engine,
    output_dir: Path,
    explain_true_on_error: bool = True,
) -> dict:
    image, input_tensor = load_image(
        image_path=image_path,
        transform=transform,
    )

    cam, predicted_class, probabilities = cam_engine.generate(
        input_tensor=input_tensor
    )

    predicted_class_name = CLASS_NAMES[predicted_class]
    confidence = probabilities[predicted_class].item()

    true_class_raw = image_path.parent.name
    true_class_name = normalize_class_name(true_class_raw)
    correct = true_class_name == predicted_class_name

    class_probabilities = {
        class_name: probabilities[index].item()
        for index, class_name in enumerate(CLASS_NAMES)
    }

    print(f"Image: {image_path.name}")
    print(f"True class: {true_class_name}")
    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Correct: {correct}")
    print("Probabilities:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {class_probabilities[class_name]:.4f}")

    class_output_dir = output_dir / true_class_raw
    original_path, heatmap_path, overlay_path = save_gradcam_outputs(
        image=image,
        cam=cam,
        output_dir=class_output_dir,
        image_stem=image_path.stem,
    )

    print(f"Heatmap saved: {heatmap_path}")
    print(f"Overlay saved: {overlay_path}")

    # Generate counterfactual CAM for the true class when prediction is wrong
    true_class_overlay_path = ""
    if explain_true_on_error and not correct and true_class_name in CLASS_NAMES:
        true_class_idx = CLASS_NAMES.index(true_class_name)
        true_cam, _, _ = cam_engine.generate(
            input_tensor=input_tensor,
            target_class=true_class_idx,
        )
        _, _, true_overlay = save_gradcam_outputs(
            image=image,
            cam=true_cam,
            output_dir=class_output_dir,
            image_stem=f"{image_path.stem}_true_class",
        )
        true_class_overlay_path = str(true_overlay)
        print(f"True-class overlay saved: {true_class_overlay_path}")

    print("-" * 60)

    return {
        "image_path": image_path,
        "true_class": true_class_name,
        "predicted_class": predicted_class_name,
        "confidence": confidence,
        "correct": correct,
        "probabilities": class_probabilities,
        "original_path": original_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "true_class_overlay_path": true_class_overlay_path,
    }


# ---------------------------------------------------------
# CSV Formatting
# ---------------------------------------------------------

def get_csv_fieldnames() -> list[str]:
    fields = [
        "image",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
    ]

    for class_name in CLASS_NAMES:
        fields.append(f"prob_{class_name}")

    fields.extend([
        "original_path",
        "heatmap_path",
        "overlay_path",
        "true_class_overlay_path",
    ])

    return fields


def create_csv_row(result: dict) -> dict:
    row = {
        "image": result["image_path"].name,
        "true_class": result["true_class"],
        "predicted_class": result["predicted_class"],
        "confidence": f"{result['confidence']:.6f}",
        "correct": result["correct"],
        "original_path": str(result["original_path"]),
        "heatmap_path": str(result["heatmap_path"]),
        "overlay_path": str(result["overlay_path"]),
        "true_class_overlay_path": result["true_class_overlay_path"],
    }

    for class_name in CLASS_NAMES:
        row[f"prob_{class_name}"] = f"{result['probabilities'][class_name]:.6f}"

    return row


# ---------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Grad-CAM or Grad-CAM++ on diabetic retinopathy samples."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Checkpoint folder and model prefix name.",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Which checkpoint file to load.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gradcam", "gradcam++"],
        default="gradcam",
        help="CAM algorithm to use.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=["layer4", "layer3"],
        default="layer4",
        help="Target residual block (layer3 gives 14x14 spatial resolution).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional square resize (e.g. 224) if input images are not pre-resized.",
    )
    parser.add_argument(
        "--no-counterfactual",
        action="store_true",
        help="Skip generating true-class CAM overlays for misclassified images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    checkpoint_filename = (
        f"{args.model_name}_best_model.pth"
        if args.checkpoint_type == "best"
        else f"{args.model_name}_last_checkpoint.pth"
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "modules"
        / "diabetic_retinopathy"
        / "artifacts"
        / "checkpoints"
        / args.model_name
        / checkpoint_filename
    )

    output_dir = (
        PROJECT_ROOT
        / "modules"
        / "diabetic_retinopathy"
        / "artifacts"
        / "explainability"
        / "gradcam"
    )

    log_path = output_dir / "gradcam_results.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    if not SAMPLES_DIR.exists():
        raise FileNotFoundError(f"Samples directory not found:\n{SAMPLES_DIR}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model: {args.model_name} ({args.checkpoint_type})")
    print(
        f"Method: {args.method} | Target layer: {args.layer} | Resize: {args.resize}")
    print(f"Samples directory: {SAMPLES_DIR}\n")

    model = create_model(checkpoint_path)
    transform = create_transform(resize=args.resize)
    image_paths = find_images(SAMPLES_DIR)

    print(f"Found {len(image_paths)} images.\n")

    csv_fields = get_csv_fieldnames()

    with create_cam_engine(model, args.method, args.layer) as cam_engine:
        with open(log_path, "w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=csv_fields)
            writer.writeheader()

            for image_path in image_paths:
                result = explain_image(
                    image_path=image_path,
                    transform=transform,
                    cam_engine=cam_engine,
                    output_dir=output_dir,
                    explain_true_on_error=not args.no_counterfactual,
                )
                row = create_csv_row(result)
                writer.writerow(row)

    print("\n" + "=" * 60)
    print("Grad-CAM processing completed.")
    print(f"Processed images: {len(image_paths)}")
    print(f"CSV log: {log_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
