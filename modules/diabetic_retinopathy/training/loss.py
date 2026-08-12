import torch
import torch.nn as nn


def create_class_weights(
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Calculate inverse-frequency class weights.

    Args:
        labels:
            Tensor containing integer class labels.

    Returns:
        Tensor containing one weight for each class.
    """

    class_counts = torch.bincount(labels)

    weights = len(labels) / (
        len(class_counts) * class_counts
    )

    return weights.float()


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Focal Loss reduces the contribution of easy examples
    and focuses training on difficult examples.

    Args:
        gamma:
            Focusing parameter. A higher value gives more
            emphasis to difficult examples.
    """

    def __init__(
        self,
        gamma: float = 2.0
    ):
        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        cross_entropy = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        probabilities = torch.exp(-cross_entropy)

        focal_loss = (
            (1 - probabilities) ** self.gamma
            * cross_entropy
        )

        return focal_loss.mean()


def create_loss(
    gamma: float = 2.0
) -> FocalLoss:
    """
    Create a Focal Loss instance.

    Args:
        gamma:
            Focusing parameter.

    Returns:
        Configured FocalLoss instance.
    """

    return FocalLoss(
        gamma=gamma
    )