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

MODEL_NAME = "ResNet50_layer3_layer4_ft_focal"
CHECKPOINT_TYPE = "best"

if CHECKPOINT_TYPE == "best":

    checkpoint_name = (
        f"{MODEL_NAME}_best_model.pth"
    )

elif CHECKPOINT_TYPE == "last":

    checkpoint_name = (
        f"{MODEL_NAME}_last_checkpoint.pth"
    )

else:

    raise ValueError(
        "CHECKPOINT_TYPE must be 'best' or 'last'."
    )

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
    / checkpoint_name
)

confusion_matrix_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "plots"
    / MODEL_NAME
    / f"{MODEL_NAME}_{CHECKPOINT_TYPE}_confusion_matrix.png"
)

metrics_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "metrics"
    / MODEL_NAME
    / f"{MODEL_NAME}_{CHECKPOINT_TYPE}_test_metrics.json"
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

print(f"Checkpoint: {checkpoint_path}")
print(f"Exists: {checkpoint_path.exists()}")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

if CHECKPOINT_TYPE == "best":

    model.load_state_dict(
        checkpoint
    )

else:

    model.load_state_dict(
        checkpoint["model_state_dict"]
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
    checkpoint_type=CHECKPOINT_TYPE,
    class_names=CLASS_NAMES,
    confusion_matrix_path=confusion_matrix_path,
    metrics_path=metrics_path
)
