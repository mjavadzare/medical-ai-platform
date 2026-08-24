from pathlib import Path
from collections import Counter

import csv


# =========================================================
# Configuration
# =========================================================

HIGH_CONFIDENCE_THRESHOLD = 0.80


# =========================================================
# Project Root
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)


# =========================================================
# Paths
# =========================================================

GRADCAM_DIR = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "explainability"
    / "gradcam"
)


LOG_PATH = (
    GRADCAM_DIR
    / "gradcam_results.csv"
)


ANALYSIS_DIR = (
    GRADCAM_DIR
    / "analysis"
)


HIGH_CONFIDENCE_ERRORS_PATH = (
    ANALYSIS_DIR
    / "high_confidence_errors.csv"
)


ALL_ERRORS_PATH = (
    ANALYSIS_DIR
    / "all_errors.csv"
)


# =========================================================
# Load CSV
# =========================================================

def load_results():

    if not LOG_PATH.exists():

        raise FileNotFoundError(
            f"Grad-CAM log not found:\n"
            f"{LOG_PATH}"
        )

    with open(
        LOG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        results = list(reader)

    return results


# =========================================================
# Convert CSV values
# =========================================================

def prepare_results(
    results
):

    for result in results:

        result["confidence"] = float(
            result["confidence"]
        )

        result["correct"] = (
            result["correct"]
            .strip()
            .lower()
            == "true"
        )

    return results


# =========================================================
# Basic statistics
# =========================================================

def calculate_basic_statistics(
    results
):

    total = len(results)

    correct = sum(
        result["correct"]
        for result in results
    )

    incorrect = (
        total - correct
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
    }


# =========================================================
# Confidence groups
# =========================================================

def get_confidence_groups(
    results
):

    high_confidence_errors = [
        result
        for result in results
        if (
            not result["correct"]
            and result["confidence"]
            >= HIGH_CONFIDENCE_THRESHOLD
        )
    ]

    low_confidence_errors = [
        result
        for result in results
        if (
            not result["correct"]
            and result["confidence"]
            < HIGH_CONFIDENCE_THRESHOLD
        )
    ]

    return (
        high_confidence_errors,
        low_confidence_errors
    )


# =========================================================
# Error statistics by true class
# =========================================================

def calculate_errors_by_true_class(
    results
):

    class_total = Counter()
    class_errors = Counter()

    for result in results:

        true_class = (
            result["true_class"]
        )

        class_total[
            true_class
        ] += 1

        if not result["correct"]:

            class_errors[
                true_class
            ] += 1

    return (
        class_total,
        class_errors
    )


# =========================================================
# Error statistics by predicted class
# =========================================================

def calculate_errors_by_predicted_class(
    results
):

    predicted_total = Counter()
    predicted_errors = Counter()

    for result in results:

        predicted_class = (
            result["predicted_class"]
        )

        predicted_total[
            predicted_class
        ] += 1

        if not result["correct"]:

            predicted_errors[
                predicted_class
            ] += 1

    return (
        predicted_total,
        predicted_errors
    )


# =========================================================
# Confusion pairs
# =========================================================

def calculate_confusion_pairs(
    results
):

    confusion_pairs = Counter()

    for result in results:

        if result["correct"]:
            continue

        pair = (
            result["true_class"],
            result["predicted_class"]
        )

        confusion_pairs[
            pair
        ] += 1

    return confusion_pairs


# =========================================================
# Save error CSV
# =========================================================

def save_error_csv(
    results,
    output_path
):

    errors = [
        result
        for result in results
        if not result["correct"]
    ]

    if not errors:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        errors[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            errors
        )


# =========================================================
# Save high-confidence errors
# =========================================================

def save_high_confidence_errors(
    results
):

    errors = [
        result
        for result in results
        if (
            not result["correct"]
            and result["confidence"]
            >= HIGH_CONFIDENCE_THRESHOLD
        )
    ]

    if not errors:

        return

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        errors[0].keys()
    )

    with open(
        HIGH_CONFIDENCE_ERRORS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            errors
        )


# =========================================================
# Print basic statistics
# =========================================================

def print_basic_statistics(
    statistics
):

    print()
    print("=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)

    print(
        f"Total samples: "
        f"{statistics['total']}"
    )

    print(
        f"Correct predictions: "
        f"{statistics['correct']}"
    )

    print(
        f"Incorrect predictions: "
        f"{statistics['incorrect']}"
    )

    print(
        f"Sample accuracy: "
        f"{statistics['accuracy']:.4f}"
    )


# =========================================================
# Print confidence analysis
# =========================================================

def print_confidence_analysis(
    high_confidence_errors,
    low_confidence_errors
):

    print()
    print("=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    print(
        f"High-confidence errors "
        f"(>= {HIGH_CONFIDENCE_THRESHOLD:.2f}): "
        f"{len(high_confidence_errors)}"
    )

    print(
        f"Low-confidence errors "
        f"(< {HIGH_CONFIDENCE_THRESHOLD:.2f}): "
        f"{len(low_confidence_errors)}"
    )


# =========================================================
# Print high-confidence errors
# =========================================================

def print_high_confidence_errors(
    errors
):

    print()
    print("=" * 70)
    print("HIGH-CONFIDENCE ERRORS")
    print("=" * 70)

    if not errors:

        print(
            "No high-confidence errors found."
        )

        return

    # Sort from highest confidence
    errors = sorted(
        errors,
        key=lambda result:
            result["confidence"],
        reverse=True
    )

    for index, result in enumerate(
        errors,
        start=1
    ):

        print()
        print(
            f"{index}. "
            f"{result['image']}"
        )

        print(
            f"   True: "
            f"{result['true_class']}"
        )

        print(
            f"   Predicted: "
            f"{result['predicted_class']}"
        )

        print(
            f"   Confidence: "
            f"{result['confidence']:.4f}"
        )

        print(
            f"   Overlay: "
            f"{result['overlay_path']}"
        )


# =========================================================
# Print errors by true class
# =========================================================

def print_errors_by_true_class(
    results
):

    (
        class_total,
        class_errors
    ) = calculate_errors_by_true_class(
        results
    )

    print()
    print("=" * 70)
    print("ERRORS BY TRUE CLASS")
    print("=" * 70)

    for class_name in sorted(
        class_total.keys()
    ):

        total = class_total[
            class_name
        ]

        errors = class_errors[
            class_name
        ]

        error_rate = (
            errors / total
            if total > 0
            else 0.0
        )

        print(
            f"{class_name}: "
            f"{errors}/{total} errors "
            f"({error_rate:.2%})"
        )


# =========================================================
# Print errors by predicted class
# =========================================================

def print_errors_by_predicted_class(
    results
):

    (
        predicted_total,
        predicted_errors
    ) = calculate_errors_by_predicted_class(
        results
    )

    print()
    print("=" * 70)
    print("ERRORS BY PREDICTED CLASS")
    print("=" * 70)

    for class_name in sorted(
        predicted_total.keys()
    ):

        total = predicted_total[
            class_name
        ]

        errors = predicted_errors[
            class_name
        ]

        print(
            f"{class_name}: "
            f"{errors} errors "
            f"out of {total} predictions"
        )


# =========================================================
# Print confusion pairs
# =========================================================

def print_confusion_pairs(
    results
):

    confusion_pairs = (
        calculate_confusion_pairs(
            results
        )
    )

    print()
    print("=" * 70)
    print("MOST COMMON CONFUSION PAIRS")
    print("=" * 70)

    if not confusion_pairs:

        print(
            "No incorrect predictions."
        )

        return

    for (
        true_class,
        predicted_class
    ), count in confusion_pairs.most_common():

        print(
            f"{true_class} -> "
            f"{predicted_class}: "
            f"{count}"
        )


# =========================================================
# Print most uncertain predictions
# =========================================================

def print_low_confidence_predictions(
    results,
    limit=10
):

    sorted_results = sorted(
        results,
        key=lambda result:
            result["confidence"]
    )

    print()
    print("=" * 70)
    print(
        f"LOWEST-CONFIDENCE PREDICTIONS "
        f"(TOP {limit})"
    )
    print("=" * 70)

    for index, result in enumerate(
        sorted_results[:limit],
        start=1
    ):

        print()
        print(
            f"{index}. "
            f"{result['image']}"
        )

        print(
            f"   True: "
            f"{result['true_class']}"
        )

        print(
            f"   Predicted: "
            f"{result['predicted_class']}"
        )

        print(
            f"   Confidence: "
            f"{result['confidence']:.4f}"
        )

        print(
            f"   Correct: "
            f"{result['correct']}"
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print(
        "Loading Grad-CAM results..."
    )

    results = load_results()

    results = prepare_results(
        results
    )

    print(
        f"Loaded {len(results)} results."
    )

    # -----------------------------------------------------
    # Create analysis directory
    # -----------------------------------------------------

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Basic statistics
    # -----------------------------------------------------

    statistics = (
        calculate_basic_statistics(
            results
        )
    )

    print_basic_statistics(
        statistics
    )

    # -----------------------------------------------------
    # Confidence groups
    # -----------------------------------------------------

    (
        high_confidence_errors,
        low_confidence_errors
    ) = get_confidence_groups(
        results
    )

    print_confidence_analysis(
        high_confidence_errors,
        low_confidence_errors
    )

    # -----------------------------------------------------
    # High-confidence errors
    # -----------------------------------------------------

    print_high_confidence_errors(
        high_confidence_errors
    )

    # -----------------------------------------------------
    # Errors by true class
    # -----------------------------------------------------

    print_errors_by_true_class(
        results
    )

    # -----------------------------------------------------
    # Errors by predicted class
    # -----------------------------------------------------

    print_errors_by_predicted_class(
        results
    )

    # -----------------------------------------------------
    # Confusion pairs
    # -----------------------------------------------------

    print_confusion_pairs(
        results
    )

    # -----------------------------------------------------
    # Lowest-confidence predictions
    # -----------------------------------------------------

    print_low_confidence_predictions(
        results
    )

    # -----------------------------------------------------
    # Save all errors
    # -----------------------------------------------------

    save_error_csv(
        results,
        ALL_ERRORS_PATH
    )

    # -----------------------------------------------------
    # Save high-confidence errors
    # -----------------------------------------------------

    save_high_confidence_errors(
        results
    )

    # -----------------------------------------------------
    # Final information
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"Input CSV:\n"
        f"{LOG_PATH}"
    )

    print(
        f"\nAnalysis directory:\n"
        f"{ANALYSIS_DIR}"
    )

    print(
        f"\nAll errors:\n"
        f"{ALL_ERRORS_PATH}"
    )

    print(
        f"\nHigh-confidence errors:\n"
        f"{HIGH_CONFIDENCE_ERRORS_PATH}"
    )

    print("=" * 70)