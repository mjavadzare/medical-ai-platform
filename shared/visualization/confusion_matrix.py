from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    model,
    labels,
    predictions,
    class_names,
    save_path,
    figsize=(8, 6)
):
    """
    Plot and save a confusion matrix.

    Args:
        model:
            Model name used in the plot title.

        labels:
            Ground-truth class labels.

        predictions:
            Predicted class labels.

        class_names:
            Names of the classification classes.

        save_path:
            Path where the confusion matrix image will be saved.

        figsize:
            Figure size as (width, height).
    """

    cm = confusion_matrix(
        labels,
        predictions
    )

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model} - Confusion Matrix")

    plt.tight_layout()

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()