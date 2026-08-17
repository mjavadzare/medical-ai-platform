from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(num_classes: int):
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model


def create_resnet50_layer4_ft(
    num_classes: int
):
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

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

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model


def freeze_batchnorm(module):
    for layer in module.modules():

        if isinstance(layer, nn.BatchNorm2d):

            layer.eval()

            for parameter in layer.parameters():
                parameter.requires_grad = False


def create_resnet50_layer3_layer4_ft_bn_frozen(
    num_classes: int
):
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    # Freeze entire pretrained backbone
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune layer3
    for parameter in model.layer3.parameters():
        parameter.requires_grad = True

    # Fine-tune layer4
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # Replace classifier
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    # Freeze BatchNorm parameters and statistics
    freeze_batchnorm(model.layer3)
    freeze_batchnorm(model.layer4)

    return model