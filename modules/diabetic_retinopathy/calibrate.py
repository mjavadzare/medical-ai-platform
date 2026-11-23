"""Fit Temperature Scaling on the DR validation split and evaluate on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modules.diabetic_retinopathy.datasets.dataloader import create_dataloaders
from modules.diabetic_retinopathy.datasets.factory import create_datasets
from modules.diabetic_retinopathy.inference.predictor import DRPredictor
from shared.calibration.temperature_scaling import (
    CalibrationResult,
    TemperatureScaler,
    compute_calibration_metrics,
    plot_reliability_diagram,
)


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference over a DataLoader and return concatenated (logits, labels)."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate DR classifier via Temperature Scaling using project splits."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to the master dataset CSV containing 'file_path' and 'diagnosis'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ResNet50_layer3_layer4_ft_bn_frozen_focal.pth"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/calibration"),
        help="Directory to save calibration_report.json and reliability diagrams.",
    )
    return parser.parse_args()


def _print_summary(split_name: str, result: CalibrationResult) -> None:
    reduction = result.to_dict()["ece_relative_reduction_pct"]
    print(f"\n=== {split_name} Set (n={result.num_samples}) ===")
    print(
        f"ECE:   {result.before.ece:.4f} -> {result.after.ece:.4f} ({reduction:+.2f}%)"
    )
    print(f"MCE:   {result.before.mce:.4f} -> {result.after.mce:.4f}")
    print(f"NLL:   {result.before.nll:.4f} -> {result.after.nll:.4f}")
    print(f"Brier: {result.before.brier_score:.4f} -> {result.after.brier_score:.4f}")
    print(
        f"Acc:   {result.before.accuracy * 100:.2f}% -> {result.after.accuracy * 100:.2f}%"
    )


def main() -> None:
    args = parse_args()

    train_ds, val_ds, test_ds = create_datasets(
        csv_path=str(args.csv_path),
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    _, val_loader, test_loader = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=args.batch_size,
        num_workers=0,
        use_weighted_sampler=False,
    )

    predictor = DRPredictor(
        checkpoint_path=args.checkpoint,
        calibration_temperature=1.0,
    )
    predictor.load()

    try:
        if predictor.cam_engine is not None:
            predictor.cam_engine.remove_hooks()

        if predictor.model is None:
            raise RuntimeError("Failed to initialize DRPredictor model.")

        print(
            f"Collecting validation logits ({len(val_ds)} samples) on {predictor.device}..."
        )
        val_logits, val_labels = collect_logits(
            predictor.model, val_loader, predictor.device
        )

        print(
            f"Collecting test logits ({len(test_ds)} samples) on {predictor.device}..."
        )
        test_logits, test_labels = collect_logits(
            predictor.model, test_loader, predictor.device
        )
    finally:
        predictor.unload()

    # Fit optimal temperature strictly on validation logits
    scaler = TemperatureScaler(init_temperature=1.0)
    val_result = scaler.fit(val_logits, val_labels, n_bins=args.n_bins)

    # Evaluate the exact same learned temperature on held-out test logits
    test_before = compute_calibration_metrics(
        test_logits, test_labels, temperature=1.0, n_bins=args.n_bins
    )
    test_after = compute_calibration_metrics(
        test_logits, test_labels, temperature=val_result.temperature, n_bins=args.n_bins
    )
    test_result = CalibrationResult(
        temperature=val_result.temperature,
        num_samples=int(test_labels.numel()),
        num_bins=args.n_bins,
        before=test_before,
        after=test_after,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "calibration_report.json"
    val_plot_path = args.output_dir / "reliability_diagram_val.png"
    test_plot_path = args.output_dir / "reliability_diagram_test.png"

    report_payload = {
        "model_name": predictor.model_name,
        "optimal_temperature": round(val_result.temperature, 6),
        "validation": val_result.to_dict(),
        "test": test_result.to_dict(),
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    plot_reliability_diagram(
        logits=val_logits,
        labels=val_labels,
        result=val_result,
        output_path=val_plot_path,
        model_name=f"{predictor.model_name} [Validation]",
    )
    plot_reliability_diagram(
        logits=test_logits,
        labels=test_labels,
        result=test_result,
        output_path=test_plot_path,
        model_name=f"{predictor.model_name} [Test]",
    )

    print(
        f"\nOptimal Temperature (fitted on Validation): T = {val_result.temperature:.4f}"
    )
    _print_summary("Validation", val_result)
    _print_summary("Test", test_result)
    print(f"\nSaved report:        {report_path}")
    print(f"Saved val diagram:   {val_plot_path}")
    print(f"Saved test diagram:  {test_plot_path}")
    print(
        "\nSet in API / docker-compose.yml:\n"
        f"  CALIBRATION_TEMPERATURE={val_result.temperature:.4f}"
    )


if __name__ == "__main__":
    main()
