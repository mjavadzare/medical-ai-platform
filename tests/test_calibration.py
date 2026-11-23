"""Unit tests for Temperature Scaling and calibration metrics."""

from pathlib import Path

import torch

from shared.calibration.temperature_scaling import (
    TemperatureScaler,
    compute_calibration_metrics,
    plot_reliability_diagram,
)


def test_temperature_scaling_reduces_nll_and_preserves_accuracy(tmp_path: Path) -> None:
    torch.manual_seed(42)
    n_samples, n_classes = 300, 5

    # Generate synthetically overconfident logits (scaled by 3.5x)
    base_logits = torch.randn(n_samples, n_classes)
    probs = torch.softmax(base_logits, dim=1)
    labels = torch.multinomial(probs, num_samples=1).squeeze(1)
    overconfident_logits = base_logits * 3.5

    scaler = TemperatureScaler()
    result = scaler.fit(overconfident_logits, labels, n_bins=15)

    assert result.temperature > 1.5
    assert result.after.nll < result.before.nll
    assert result.after.ece < result.before.ece
    assert result.after.accuracy == result.before.accuracy

    out_png = plot_reliability_diagram(
        overconfident_logits,
        labels,
        result,
        output_path=tmp_path / "reliability.png",
    )
    assert out_png.exists() and out_png.stat().st_size > 0


def test_compute_metrics_accepts_custom_temperature() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0], [1.0, 3.0]])
    labels = torch.tensor([0, 1, 0, 1])

    metrics = compute_calibration_metrics(logits, labels, temperature=1.0, n_bins=10)
    assert metrics.accuracy == 1.0
    assert 0.0 <= metrics.ece <= 1.0
