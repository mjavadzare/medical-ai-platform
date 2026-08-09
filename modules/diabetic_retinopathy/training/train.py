from pathlib import Path
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.diabetic_retinopathy.datasets.dataloader import (
    create_dataloaders
)
from modules.diabetic_retinopathy.models.resnet import (
    create_resnet50
)
from modules.diabetic_retinopathy.training.engine import (
    train_one_epoch,
    validate_one_epoch
)
from modules.diabetic_retinopathy.training.loss import (
    create_class_weights,
    create_loss
)
from modules.diabetic_retinopathy.datasets.factory import (
    create_datasets
)


# -------------------------
# Paths
# -------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[3]
print(f"Project Root:{PROJECT_ROOT}")

csv_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diabetic_retinopathy"
    / "image_metadata.csv"
)

best_checkpoint_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "checkpoints"
    / "resnet50_best_checkpoint.pth"
)

last_checkpoint_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "checkpoints"
    / "resnet50_last_checkpoint.pth"
)

# Change image relative path to absolute path
df = pd.read_csv(csv_path)
df["file_path"] = df["file_path"].apply(
    lambda path: (PROJECT_ROOT / Path(path)).resolve()
)

df["file_path"] = df["file_path"].astype(str)

df.to_csv(csv_path)

# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



# -------------------------
# Create Dataset
# -------------------------

train_dataset, val_dataset, test_dataset = create_datasets(
    csv_path=csv_path
)

# -------------------------
# Create DataLoader
# -------------------------

train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=32,
    num_workers=0
)
labels = torch.tensor(
    train_dataset.dataframe["diagnosis"].values,
    dtype=torch.long
)

# -------------------------
# Loss
# -------------------------

class_weights = create_class_weights(labels)

criterion = create_loss(
    class_weights=class_weights.to(device)
)


# -------------------------
# Model (ResNet50)
# -------------------------

model = create_resnet50(
    num_classes=5
)

model = model.to(device)

# -------------------------
# Optimizer
# -------------------------

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.001
)


# -------------------------
# Scheduler
# -------------------------

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2,
    min_lr=1e-6
)



num_epochs = 20
best_val_loss = float("inf")
start_epoch = 0


# -------------------------
# Resume Training
# -------------------------

if last_checkpoint_path.exists():

    print("Loading last checkpoint...")

    checkpoint = torch.load(
        last_checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    print(
        f"Resuming from epoch {start_epoch + 1}"
    )


# -------------------------
# Training
# -------------------------

print("Training lopp will be begin.")

for epoch in range(start_epoch, num_epochs):

    # -------------------------
    # Train
    # -------------------------

    train_loss, train_accuracy = train_one_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    # -------------------------
    # Validation
    # -------------------------

    val_loss, val_accuracy = validate_one_epoch(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler.step(val_loss)

    # -------------------------
    # Learning rate
    # -------------------------

    current_lr = optimizer.param_groups[0]["lr"]

    # -------------------------
    # Metrics
    # -------------------------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"| "
        f"Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| "
        f"Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.4f} "
        f"| "
        f"LR: {current_lr:.2e}"
    )

    # -------------------------
    # Save best model
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            best_checkpoint_path
        )

        print("✓ Best model saved")

    # -------------------------
    # Save last checkpoint
    # -------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        last_checkpoint_path
    )