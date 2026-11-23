"""Post-hoc probability calibration via Temperature Scaling and reliability analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CalibrationMetrics:
    """Summary of calibration and accuracy metrics."""

    ece: float
    mce: float
    nll: float
    brier_score: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationResult:
    """Before/after calibration comparison and optimal temperature."""

    temperature: float
    num_samples: int
    num_bins: int
    before: CalibrationMetrics
    after: CalibrationMetrics

    def to_dict(self) -> dict:
        return {
            "temperature": round(self.temperature, 6),
            "num_samples": self.num_samples,
            "num_bins": self.num_bins,
            "before": asdict(self.before),
            "after": asdict(self.after),
            "ece_relative_reduction_pct": round(
                (1.0 - self.after.ece / max(self.before.ece, 1e-12)) * 100.0, 2
            ),
        }


def _compute_bin_stats(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bin accuracy, confidence, counts, and bin edges."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins, dtype=np.float64)
    bin_confs = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        low = bin_boundaries[i].item()
        high = bin_boundaries[i + 1].item()
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)

        count = int(in_bin.sum().item())
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = accuracies[in_bin].mean().item()
            bin_confs[i] = confidences[in_bin].mean().item()

    return bin_accs, bin_confs, bin_counts, bin_boundaries.numpy()


def compute_calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    n_bins: int = 15,
) -> CalibrationMetrics:
    """Compute ECE, MCE, NLL, multiclass Brier score, and accuracy."""
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()

    scaled_logits = logits / float(temperature)
    probs = F.softmax(scaled_logits, dim=1)

    nll = F.cross_entropy(scaled_logits, labels).item()

    num_classes = logits.shape[1]
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    brier_score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()

    accuracy = probs.argmax(dim=1).eq(labels).float().mean().item()

    bin_accs, bin_confs, bin_counts, _ = _compute_bin_stats(
        probs, labels, n_bins=n_bins
    )
    total = max(int(bin_counts.sum()), 1)
    gaps = np.abs(bin_accs - bin_confs)
    weights = bin_counts / total

    ece = float(np.sum(weights * gaps))
    mce = float(np.max(gaps[bin_counts > 0])) if np.any(bin_counts > 0) else 0.0

    return CalibrationMetrics(
        ece=round(ece, 6),
        mce=round(mce, 6),
        nll=round(nll, 6),
        brier_score=round(brier_score, 6),
        accuracy=round(accuracy, 6),
    )


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling fitted by minimizing NLL on validation logits."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        init_log_t = float(np.log(max(init_temperature, 1e-4)))
        self.log_temperature = nn.Parameter(
            torch.tensor([init_log_t], dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).detach().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_temperature)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
        max_iter: int = 100,
    ) -> CalibrationResult:
        """Optimize scalar temperature on validation logits using L-BFGS."""
        logits = logits.detach().float().cpu()
        labels = labels.detach().long().cpu()

        before = compute_calibration_metrics(
            logits, labels, temperature=1.0, n_bins=n_bins
        )

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=0.05,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)

        opt_temp = self.temperature
        after = compute_calibration_metrics(
            logits, labels, temperature=opt_temp, n_bins=n_bins
        )

        return CalibrationResult(
            temperature=opt_temp,
            num_samples=int(labels.numel()),
            num_bins=n_bins,
            before=before,
            after=after,
        )


def plot_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    result: CalibrationResult,
    output_path: str | Path,
    model_name: str = "ResNet50 (Focal Loss)",
) -> Path:
    """Generate side-by-side Before/After Reliability Diagrams with calibration gaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()

    probs_before = F.softmax(logits, dim=1)
    probs_after = F.softmax(logits / result.temperature, dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, dpi=200)
    panels = [
        ("Before Calibration (T = 1.000)", probs_before, result.before),
        (
            f"After Temperature Scaling (T = {result.temperature:.3f})",
            probs_after,
            result.after,
        ),
    ]

    for ax, (title, probs, metrics) in zip(axes, panels, strict=True):
        bin_accs, bin_confs, bin_counts, edges = _compute_bin_stats(
            probs, labels, n_bins=result.num_bins
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = (edges[1] - edges[0]) * 0.88
        non_empty = bin_counts > 0

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#444444",
            linewidth=1.2,
            label="Perfect",
        )

        ax.bar(
            centers[non_empty],
            bin_accs[non_empty],
            width=width,
            color="#285A48",
            alpha=0.85,
            edgecolor="#091413",
            linewidth=0.7,
            label="Accuracy",
        )

        gap_bottoms = np.minimum(bin_accs[non_empty], bin_confs[non_empty])
        gap_heights = np.abs(bin_confs[non_empty] - bin_accs[non_empty])
        ax.bar(
            centers[non_empty],
            gap_heights,
            bottom=gap_bottoms,
            width=width,
            color="#D9534F",
            alpha=0.35,
            edgecolor="#B52B27",
            hatch="//",
            linewidth=0.7,
            label="Calibration Gap",
        )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Confidence", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.45)

        summary_text = (
            f"ECE:   {metrics.ece:.4f}\n"
            f"MCE:   {metrics.mce:.4f}\n"
            f"NLL:   {metrics.nll:.4f}\n"
            f"Brier: {metrics.brier_score:.4f}\n"
            f"Acc:   {metrics.accuracy * 100:.2f}%"
        )
        ax.text(
            0.04,
            0.96,
            summary_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8.5,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#F7FAF9",
                "edgecolor": "#408A71",
            },
        )
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    axes[0].set_ylabel("Empirical Accuracy", fontsize=10)
    fig.suptitle(
        f"Reliability Diagram — {model_name} (n={result.num_samples})",
        fontsize=12.5,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
