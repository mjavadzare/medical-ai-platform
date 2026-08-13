import torch

from torch import cuda
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler
)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
):
    pin_memory = cuda.is_available()

    # -------------------------
    # Train sampler
    # -------------------------

    if use_weighted_sampler:

        labels = torch.tensor(
            train_dataset.dataframe["diagnosis"].values,
            dtype=torch.long
        )

        class_counts = torch.bincount(labels)

        class_weights = 1.0 / class_counts.float()

        sample_weights = class_weights[labels]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # -------------------------
    # Validation
    # -------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # -------------------------
    # Test
    # -------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )