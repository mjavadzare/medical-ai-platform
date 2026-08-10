import torch


def predict(
    model,
    data_loader,
    device
):
    """
    Generate predictions for a classification model.

    Args:
        model:
            PyTorch classification model.

        data_loader:
            DataLoader containing input images and labels.

        device:
            Device used for inference.

    Returns:
        tuple[list[int], list[int]]:
            Ground-truth labels and model predictions.
    """

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in data_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    return all_labels, all_predictions