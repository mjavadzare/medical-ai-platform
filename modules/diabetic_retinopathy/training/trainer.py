from pathlib import Path

import torch

from modules.diabetic_retinopathy.training.engine import (
    train_one_epoch,
    validate_one_epoch
)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs: int,
    checkpoint_path,
    last_checkpoint_path,
    resume=True
):
    checkpoint_path = Path(checkpoint_path)
    last_checkpoint_path = Path(last_checkpoint_path)

    # -------------------------
    # Create checkpoint directories
    # -------------------------

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    last_checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------
    # Initial state
    # -------------------------

    start_epoch = 0
    best_val_macro_f1 = 0.0

    # -------------------------
    # Resume training
    # -------------------------

    if resume and last_checkpoint_path.exists():

        print(
            f"Loading checkpoint: "
            f"{last_checkpoint_path}"
        )

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

        start_epoch = checkpoint["epoch"]

        best_val_macro_f1 = checkpoint[
            "best_val_macro_f1"
        ]

        print(
            f"Resuming training from epoch "
            f"{start_epoch + 1}"
        )

        print(
            f"Best validation Macro F1: "
            f"{best_val_macro_f1:.4f}"
        )

    else:

        print("Starting training from scratch.")

    # -------------------------
    # Training
    # -------------------------

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

        (
            val_loss,
            val_accuracy,
            val_macro_f1
        ) = validate_one_epoch(
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
            f"| Val Macro F1: {val_macro_f1:.4f} "
            f"| LR: {current_lr:.2e}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if val_macro_f1 > best_val_macro_f1:

            best_val_macro_f1 = val_macro_f1

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                "✓ Best model saved "
                f"(Macro F1: {best_val_macro_f1:.4f})"
            )

        # -------------------------
        # Save last checkpoint
        # -------------------------

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
            },
            last_checkpoint_path
        )

        print("✓ Last checkpoint saved")