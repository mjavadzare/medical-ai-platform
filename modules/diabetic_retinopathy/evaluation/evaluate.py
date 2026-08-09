from pathlib import Path
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from modules.diabetic_retinopathy.models.resnet import (
    create_resnet50
)
from modules.diabetic_retinopathy.datasets.factory import (
    create_datasets
)
from modules.diabetic_retinopathy.datasets.dataloader import (
    create_dataloaders
)
from shared.visualization.confusion_matrix import (
    plot_confusion_matrix
)
from shared.metrics.classification import (
    calculate_classification_metrics,
    save_classification_metrics
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
print(f"Project Root:{PROJECT_ROOT}")

csv_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diabetic_retinopathy"
    / "image_metadata.csv"
)

best_checkpoint_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "checkpoints"
    / "resnet50_best_checkpoint.pth"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = create_resnet50(num_classes=5)

model.load_state_dict(
    torch.load(
        best_checkpoint_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()


train_dataset, val_dataset, test_dataset = create_datasets(
    csv_path=csv_path
)



_, _, test_loader = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=32,
    num_workers=0
)



all_predictions = []
all_labels = []

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )




# Metrics
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print(f"Test Accuracy: {accuracy:.4f}")

print(
    classification_report(
        all_labels,
        all_predictions
    )
)

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# ResNet50 Confusion Matrix
confusion_matrix_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "plots"
    / "resnet50"
    / "confusion_matrix.png"
)

plot_confusion_matrix(
    model="ResNet50",
    labels=all_labels,
    predictions=all_predictions,
    class_names=class_names,
    save_path=confusion_matrix_path
)


# ResNet50 Metrics
metrics = calculate_classification_metrics(
    labels=all_labels,
    predictions=all_predictions
)

metrics_path = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "metrics"
    / "resnet50"
    / "test_metrics.json"
)

save_classification_metrics(
    metrics=metrics,
    save_path=metrics_path
)