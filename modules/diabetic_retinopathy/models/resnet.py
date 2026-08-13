from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(num_classes: int):
    # Load pretrained ResNet50
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    # Freeze the pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model


def create_resnet50_layer4_ft(num_classes: int):
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model


def create_resnet50_layer3_layer4_ft(
    num_classes: int
):
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3
    for param in model.layer3.parameters():
        param.requires_grad = True

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model