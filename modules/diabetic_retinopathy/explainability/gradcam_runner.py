from pathlib import Path
import csv

import torch

from PIL import Image

from torchvision import transforms

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from shared.explainability.gradcam import (
    GradCAM
)

from shared.explainability.visualization import (
    save_gradcam_outputs
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = (
    "ResNet50_layer3_layer4_ft_bn_frozen_focal"
)

NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


# ---------------------------------------------------------
# Class name normalization
# ---------------------------------------------------------

CLASS_NAME_MAPPING = {
    "No_DR": "No DR",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "Proliferate_DR": "Proliferative DR",
}


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "checkpoints"
    / MODEL_NAME
    / f"{MODEL_NAME}_best_model.pth"
)


SAMPLES_DIR = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "diabetic_retinopathy"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "explainability"
    / "gradcam"
)


LOG_PATH = (
    OUTPUT_DIR
    / "gradcam_results.csv"
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def create_model():

    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=NUM_CLASSES
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def create_transform():

    weights = (
        ResNet50_Weights.DEFAULT
    )

    return transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std
        )
    ])


# ---------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------

def create_gradcam(model):

    """
    Create Grad-CAM using the final
    convolutional layer of ResNet50.
    """

    target_layer = (
        model.layer4[-1].conv3
    )

    return GradCAM(
        model=model,
        target_layer=target_layer
    )


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

def load_image(
    image_path,
    transform
):

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = (
        input_tensor.to(DEVICE)
    )

    return (
        image,
        input_tensor
    )


# ---------------------------------------------------------
# Normalize class name
# ---------------------------------------------------------

def normalize_class_name(
    class_name
):

    return CLASS_NAME_MAPPING.get(
        class_name,
        class_name
    )


# ---------------------------------------------------------
# Generate explanation
# ---------------------------------------------------------

def explain_image(
    image_path,
    model,
    transform,
    gradcam
):

    image, input_tensor = (
        load_image(
            image_path=image_path,
            transform=transform
        )
    )

    (
        cam,
        predicted_class,
        probabilities
    ) = gradcam.generate(
        input_tensor=input_tensor
    )

    predicted_class_name = (
        CLASS_NAMES[
            predicted_class
        ]
    )

    confidence = (
        probabilities[
            predicted_class
        ].item()
    )

    true_class_raw = (
        image_path.parent.name
    )

    true_class_name = (
        normalize_class_name(
            true_class_raw
        )
    )

    correct = (
        true_class_name
        == predicted_class_name
    )

    # -----------------------------------------------------
    # Extract probabilities
    # -----------------------------------------------------

    class_probabilities = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_probabilities[
            class_name
        ] = probabilities[index].item()

    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"True class: "
        f"{true_class_name}"
    )

    print(
        f"Predicted class: "
        f"{predicted_class_name}"
    )

    print(
        f"Confidence: "
        f"{confidence:.4f}"
    )

    print(
        f"Correct: "
        f"{correct}"
    )

    print(
        "Probabilities:"
    )

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name}: "
            f"{class_probabilities[class_name]:.4f}"
        )

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    class_output_dir = (
        OUTPUT_DIR
        / true_class_raw
    )

    (
        original_path,
        heatmap_path,
        overlay_path
    ) = save_gradcam_outputs(
        image=image,
        cam=cam,
        output_dir=class_output_dir,
        image_stem=image_path.stem
    )

    print(
        f"Heatmap saved: "
        f"{heatmap_path}"
    )

    print(
        f"Overlay saved: "
        f"{overlay_path}"
    )

    print(
        "-" * 60
    )

    return {
        "image_path": image_path,
        "true_class": true_class_name,
        "predicted_class": predicted_class_name,
        "confidence": confidence,
        "correct": correct,
        "probabilities": class_probabilities,
        "original_path": original_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
    }


# ---------------------------------------------------------
# Find images
# ---------------------------------------------------------

def find_images():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    image_paths = []

    for class_dir in (
        SAMPLES_DIR.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        for image_path in (
            class_dir.iterdir()
        ):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in image_extensions
            ):

                image_paths.append(
                    image_path
                )

    return sorted(
        image_paths
    )


# ---------------------------------------------------------
# CSV field names
# ---------------------------------------------------------

def get_csv_fieldnames():

    fields = [
        "image",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
    ]

    # Add probability column for every class
    for class_name in CLASS_NAMES:

        fields.append(
            f"prob_{class_name}"
        )

    fields.extend([
        "original_path",
        "heatmap_path",
        "overlay_path",
    ])

    return fields


# ---------------------------------------------------------
# Create CSV row
# ---------------------------------------------------------

def create_csv_row(
    result
):

    row = {
        "image": result[
            "image_path"
        ].name,

        "true_class": result[
            "true_class"
        ],

        "predicted_class": result[
            "predicted_class"
        ],

        "confidence": (
            f"{result['confidence']:.6f}"
        ),

        "correct": result[
            "correct"
        ],

        "original_path": str(
            result["original_path"]
        ),

        "heatmap_path": str(
            result["heatmap_path"]
        ),

        "overlay_path": str(
            result["overlay_path"]
        ),
    }

    for class_name in CLASS_NAMES:

        row[
            f"prob_{class_name}"
        ] = (
            f"{result['probabilities'][class_name]:.6f}"
        )

    return row


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------
    # Validate paths
    # -----------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    if not SAMPLES_DIR.exists():

        raise FileNotFoundError(
            "Samples directory not found:\n"
            f"{SAMPLES_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Information
    # -----------------------------------------------------

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Samples directory: "
        f"{SAMPLES_DIR}"
    )

    print()

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = create_model()

    transform = create_transform()

    gradcam = create_gradcam(
        model=model
    )

    # -----------------------------------------------------
    # Find images
    # -----------------------------------------------------

    image_paths = find_images()

    print(
        f"Found {len(image_paths)} images."
    )

    print()

    # -----------------------------------------------------
    # Create CSV
    # -----------------------------------------------------

    csv_fields = get_csv_fieldnames()

    with open(
        LOG_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as log_file:

        writer = csv.DictWriter(
            log_file,
            fieldnames=csv_fields
        )

        writer.writeheader()

        # -------------------------------------------------
        # Process images
        # -------------------------------------------------

        for image_path in image_paths:

            result = explain_image(
                image_path=image_path,
                model=model,
                transform=transform,
                gradcam=gradcam
            )

            row = create_csv_row(
                result
            )

            writer.writerow(
                row
            )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 60)

    print(
        "Grad-CAM processing completed."
    )

    print(
        f"Processed images: "
        f"{len(image_paths)}"
    )

    print(
        f"CSV log: "
        f"{LOG_PATH}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print("=" * 60)