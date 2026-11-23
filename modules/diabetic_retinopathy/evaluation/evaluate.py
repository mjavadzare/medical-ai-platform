import argparse
import sys
from pathlib import Path

import torch
from torchvision.models import efficientnet_b0, resnet50

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.diabetic_retinopathy.datasets.dataloader import create_dataloaders
from modules.diabetic_retinopathy.datasets.factory import create_datasets
from modules.diabetic_retinopathy.models.densenet import (
    create_densenet121_features_ft,
)
from shared.evaluation.classification import evaluate_model

# -------------------------
# Constants
# -------------------------

DEFAULT_MODEL_NAME = "ResNet50_layer3_layer4_ft_bn_frozen_focal"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


# -------------------------
# Model Registry
# -------------------------


def infer_architecture(model_name: str) -> str:
    lowered = model_name.lower()
    if "resnet50" in lowered:
        return "resnet50"
    if "densenet121" in lowered:
        return "densenet121"
    if "efficientnet" in lowered:
        return "efficientnet_b0"
    raise ValueError(
        f"Could not infer architecture from model_name '{model_name}'. "
        "Pass --arch explicitly (resnet50 | densenet121 | efficientnet_b0)."
    )


def build_model(arch: str, num_classes: int = 5) -> torch.nn.Module:
    if arch == "resnet50":
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "densenet121":
        return create_densenet121_features_ft(num_classes=num_classes)

    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
        return model

    raise ValueError(
        f"Unsupported architecture '{arch}'. "
        "Choose from: resnet50, densenet121, efficientnet_b0."
    )


def load_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    return model.to(device)


# -------------------------
# CLI
# -------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Diabetic Retinopathy classification model."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Experiment / checkpoint folder name.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["resnet50", "densenet121", "efficientnet_b0"],
        default=None,
        help="Backbone architecture (auto-inferred from --model-name if omitted).",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Checkpoint variant to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation DataLoader batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    return parser.parse_args()


# -------------------------
# Main
# -------------------------


def main():
    args = parse_args()
    arch = args.arch or infer_architecture(args.model_name)

    checkpoint_filename = (
        f"{args.model_name}_best_model.pth"
        if args.checkpoint_type == "best"
        else f"{args.model_name}_last_checkpoint.pth"
    )

    csv_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "diabetic_retinopathy"
        / "image_metadata.csv"
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

    confusion_matrix_path = (
        PROJECT_ROOT
        / "modules"
        / "diabetic_retinopathy"
        / "artifacts"
        / "plots"
        / args.model_name
        / f"{args.model_name}_{args.checkpoint_type}_confusion_matrix.png"
    )

    metrics_path = (
        PROJECT_ROOT
        / "modules"
        / "diabetic_retinopathy"
        / "artifacts"
        / "metrics"
        / args.model_name
        / f"{args.model_name}_{args.checkpoint_type}_test_metrics.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {args.model_name} | Architecture: {arch}")
    print(f"Checkpoint: {checkpoint_path}")

    model = build_model(arch=arch, num_classes=len(CLASS_NAMES))
    model = load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()

    train_dataset, val_dataset, test_dataset = create_datasets(csv_path=csv_path)

    _, _, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        model_name=args.model_name,
        checkpoint_type=args.checkpoint_type,
        class_names=CLASS_NAMES,
        confusion_matrix_path=confusion_matrix_path,
        metrics_path=metrics_path,
    )


if __name__ == "__main__":
    main()
