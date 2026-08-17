from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.diabetic_retinopathy.datasets.factory import (
    create_datasets
)

from modules.diabetic_retinopathy.datasets.dataloader import (
    create_dataloaders
)

from modules.diabetic_retinopathy.models.efficientnet import (
    create_efficientnet_b0
)

from modules.diabetic_retinopathy.training.loss import (
    create_loss
)

from modules.diabetic_retinopathy.training.trainer import (
    train_model
)


# -------------------------
# Configuration
# -------------------------

MODEL_NAME = "efficientnet_b0"

NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = 20

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3

FOCAL_GAMMA = 2.0

USE_WEIGHTED_SAMPLER = False

SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 2
MIN_LEARNING_RATE = 1e-6

RESUME = True


# -------------------------
# Project Root
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# -------------------------
# Paths
# -------------------------

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


last_checkpoint_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "checkpoints"
    / MODEL_NAME
    / f"{MODEL_NAME}_last_checkpoint.pth"
)


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")


# -------------------------
# Dataset
# -------------------------

train_dataset, val_dataset, test_dataset = create_datasets(
    csv_path=csv_path
)


# -------------------------
# DataLoader
# -------------------------

train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    use_weighted_sampler=USE_WEIGHTED_SAMPLER
)


# -------------------------
# Loss
# -------------------------

criterion = create_loss(
    gamma=FOCAL_GAMMA
)


# -------------------------
# Model
# -------------------------

model = create_efficientnet_b0(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# -------------------------
# Optimizer
# -------------------------

optimizer = AdamW(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# -------------------------
# Scheduler
# -------------------------

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE,
    min_lr=MIN_LEARNING_RATE
)


# -------------------------
# Training
# -------------------------

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=NUM_EPOCHS,
    checkpoint_path=checkpoint_path,
    last_checkpoint_path=last_checkpoint_path,
    resume=RESUME
)