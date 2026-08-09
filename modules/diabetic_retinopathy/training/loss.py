import torch


def create_class_weights(labels):

    class_counts = torch.bincount(labels)

    weights = len(labels) / (
        len(class_counts) * class_counts
    )

    return weights.float()


def create_loss(class_weights):

    return torch.nn.CrossEntropyLoss(
        weight=class_weights
    )