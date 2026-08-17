from torch import nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def create_efficientnet_b0(
    num_classes: int
):
    # Load pretrained EfficientNet-B0
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )

    return model