from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from shared.evaluation.prediction import (
    predict
)

from shared.visualization.confusion_matrix import (
    plot_confusion_matrix
)

from shared.metrics.classification import (
    calculate_classification_metrics,
    save_classification_metrics
)


def evaluate_model(
    model,
    data_loader,
    device,
    model_name,
    checkpoint_type,
    class_names,
    confusion_matrix_path: Path,
    metrics_path: Path
):
    """
    Evaluate a classification model on a dataset.

    The function generates predictions, calculates classification
    metrics, prints evaluation results, saves a confusion matrix,
    and saves classification metrics.

    Args:
        model:
            PyTorch classification model.

        data_loader:
            DataLoader containing the evaluation dataset.

        device:
            Device used for inference.

        model_name:
            Name of the model used in artifact titles and filenames.
        
        checkpoint_type:
            "last" or "best" model

        class_names:
            List of class names corresponding to class indices.

        confusion_matrix_path:
            Path where the confusion matrix image will be saved.

        metrics_path:
            Path where classification metrics will be saved.

    Returns:
        dict:
            Calculated classification metrics.
    """

    # -------------------------
    # Prediction
    # -------------------------

    all_labels, all_predictions = predict(
        model=model,
        data_loader=data_loader,
        device=device
    )

    # -------------------------
    # Basic Metrics
    # -------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print(
        f"{model_name} Test Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            zero_division=0
        )
    )

    print(
        confusion_matrix(
            all_labels,
            all_predictions
        )
    )

    # -------------------------
    # Confusion Matrix
    # -------------------------

    plot_confusion_matrix(
        model=model_name+checkpoint_type,
        labels=all_labels,
        predictions=all_predictions,
        class_names=class_names,
        save_path=confusion_matrix_path
    )

    # -------------------------
    # Classification Metrics
    # -------------------------

    metrics = calculate_classification_metrics(
        labels=all_labels,
        predictions=all_predictions
    )

    save_classification_metrics(
        metrics=metrics,
        save_path=metrics_path
    )

    return metrics