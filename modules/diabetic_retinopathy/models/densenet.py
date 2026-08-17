from torch import nn
from torchvision.models import (
    densenet121,
    DenseNet121_Weights
)


def create_densenet121(
    num_classes: int
):
    model = densenet121(
        weights=DenseNet121_Weights.DEFAULT
    )

    # Freeze all layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace classifier
    model.classifier = nn.Linear(
        in_features=model.classifier.in_features,
        out_features=num_classes
    )

    return model


def create_densenet121_features_ft(
    num_classes: int
):
    model = densenet121(
        weights=DenseNet121_Weights.DEFAULT
    )

    # Freeze everything
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the last dense block
    for parameter in model.features.denseblock4.parameters():
        parameter.requires_grad = True

    # Fine-tune the final normalization layer
    for parameter in model.features.norm5.parameters():
        parameter.requires_grad = True

    # Replace classifier
    model.classifier = nn.Linear(
        in_features=model.classifier.in_features,
        out_features=num_classes
    )

    return model