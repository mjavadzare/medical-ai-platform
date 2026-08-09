import json
from pathlib import Path

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
classification_report
)

def calculate_classification_metrics(
    labels,
    predictions
    ):
    """
    Calculate common classification metrics.

    ```
    Parameters
    ----------
    labels : array-like
        Ground-truth class labels.

    predictions : array-like
        Predicted class labels produced by the model.

    Returns
    -------
    dict
        Dictionary containing accuracy, macro/weighted precision,
        recall, F1-score, and the classification report.
    """

    metrics = {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision_macro": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "precision_weighted": precision_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "recall_macro": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "recall_weighted": recall_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "f1_macro": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "f1_weighted": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0
        )
    }

    return metrics


def save_classification_metrics(
    metrics,
    save_path
    ):
    """
    Save classification metrics to a JSON file.

    ```
    Parameters
    ----------
    metrics : dict
        Classification metrics returned by
        ``calculate_classification_metrics``.

    save_path : str or pathlib.Path
        Path where the metrics JSON file will be saved.

    Returns
    -------
    None
        Saves the metrics to the specified path.
    """

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

