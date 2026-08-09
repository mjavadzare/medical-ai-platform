import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(
    model:str,
    labels,
    predictions,
    class_names,
    save_path=None,
    figsize=(8, 6)
    ):
    """
    Plot and optionally save a confusion matrix as a heatmap.

    ```
    Parameters
    ----------
    labels : array-like
        Ground-truth class labels.

    predictions : array-like
        Predicted class labels produced by the model.

    class_names : list[str]
        Names of the classes. The order must match the
        numeric class labels used by the model.

    save_path : str or pathlib.Path, optional
        Path where the confusion matrix image will be saved.
        If None, the plot will not be saved.

    figsize : tuple, optional
        Figure size as (width, height), by default (8, 6).

    Returns
    -------
    None
        Displays the confusion matrix and optionally saves it.
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
    plt.title(f"Confusion Matrix - {model}")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close()
