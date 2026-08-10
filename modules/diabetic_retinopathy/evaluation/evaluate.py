from pathlib import Path

import torch

from modules.diabetic_retinopathy.models.resnet import (
    create_resnet50
)

from modules.diabetic_retinopathy.datasets.factory import (
    create_datasets
)

from modules.diabetic_retinopathy.datasets.dataloader import (
    create_dataloaders
)

from shared.evaluation.classification import (
    evaluate_model
)

# -------------------------
# Configuration
# -------------------------

MODEL_NAME = "resnet50_frozen"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# -------------------------
# Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    / MODEL_NAME
    / f"{MODEL_NAME}_best_model.pth"
)

confusion_matrix_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "plots"
    / MODEL_NAME
    / f"{MODEL_NAME}_confusion_matrix.png"
)

metrics_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "metrics"
    / MODEL_NAME
    / f"{MODEL_NAME}_test_metrics.json"
)




# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------
# Model
# -------------------------

model = create_resnet50(
    num_classes=5
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model = model.to(device)


# -------------------------
# Dataset
# -------------------------

train_dataset, val_dataset, test_dataset = create_datasets(
    csv_path=csv_path
)


# -------------------------
# DataLoader
# -------------------------

_, _, test_loader = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=32,
    num_workers=0
)


# -------------------------
# Evaluation
# -------------------------

evaluate_model(
    model=model,
    data_loader=test_loader,
    device=device,
    model_name=MODEL_NAME,
    class_names=CLASS_NAMES,
    confusion_matrix_path=confusion_matrix_path,
    metrics_path=metrics_path
)
