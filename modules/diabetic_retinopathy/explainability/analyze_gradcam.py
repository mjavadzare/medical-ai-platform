import csv
import html
import math
import os
from collections import Counter
from pathlib import Path

# =========================================================
# Configuration
# =========================================================

HIGH_CONFIDENCE_THRESHOLD = 0.80

CLASS_ORDER = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_ORDER)}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRADCAM_DIR = (
    PROJECT_ROOT
    / "modules"
    / "diabetic_retinopathy"
    / "artifacts"
    / "explainability"
    / "gradcam"
)
LOG_PATH = GRADCAM_DIR / "gradcam_results.csv"
ANALYSIS_DIR = GRADCAM_DIR / "analysis"

ALL_ERRORS_PATH = ANALYSIS_DIR / "all_errors.csv"
HIGH_CONFIDENCE_ERRORS_PATH = ANALYSIS_DIR / "high_confidence_errors.csv"
UNDER_DIAGNOSIS_ERRORS_PATH = ANALYSIS_DIR / "under_diagnosis_errors.csv"
HTML_REPORT_PATH = ANALYSIS_DIR / "error_report.html"


# =========================================================
# Load & Enrich Results
# =========================================================

def load_results() -> list[dict]:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Grad-CAM log not found:\n{LOG_PATH}")

    with open(LOG_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def compute_entropy(probs: list[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0)


def prepare_results(results: list[dict]) -> list[dict]:
    enriched = []
    for row in results:
        item = dict(row)
        item["confidence"] = float(item["confidence"])
        item["correct"] = item["correct"].strip().lower() == "true"

        probs = {}
        for cls_name in CLASS_ORDER:
            col = f"prob_{cls_name}"
            probs[cls_name] = float(
                item[col]) if col in item and item[col] != "" else 0.0
        item["probs"] = probs

        true_cls = item["true_class"]
        pred_cls = item["predicted_class"]

        true_idx = CLASS_TO_IDX.get(true_cls, -1)
        pred_idx = CLASS_TO_IDX.get(pred_cls, -1)

        if true_idx >= 0 and pred_idx >= 0:
            severity_diff = pred_idx - true_idx
            severity_distance = abs(severity_diff)
            if severity_diff < 0:
                error_direction = "under_diagnosis"
            elif severity_diff > 0:
                error_direction = "over_diagnosis"
            else:
                error_direction = "correct"
        else:
            severity_diff = 0
            severity_distance = 0
            error_direction = "unknown"

        true_prob = probs.get(true_cls, 0.0)
        item["severity_diff"] = severity_diff
        item["severity_distance"] = severity_distance
        item["error_direction"] = error_direction
        item["true_class_prob"] = true_prob
        item["confidence_gap"] = item["confidence"] - true_prob
        item["entropy"] = compute_entropy(list(probs.values()))

        enriched.append(item)

    return enriched


# =========================================================
# Statistics
# =========================================================

def calculate_statistics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    errors = [r for r in results if not r["correct"]]

    high_conf_errors = [
        r for r in errors if r["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
    ]
    low_conf_errors = [
        r for r in errors if r["confidence"] < HIGH_CONFIDENCE_THRESHOLD
    ]
    under_diag = [r for r in errors if r["error_direction"]
                  == "under_diagnosis"]
    over_diag = [r for r in errors if r["error_direction"] == "over_diagnosis"]
    adjacent_errors = [r for r in errors if r["severity_distance"] == 1]
    multistep_errors = [r for r in errors if r["severity_distance"] >= 2]

    return {
        "total": total,
        "correct": correct,
        "incorrect": len(errors),
        "accuracy": correct / total if total > 0 else 0.0,
        "errors": errors,
        "high_conf_errors": high_conf_errors,
        "low_conf_errors": low_conf_errors,
        "under_diag": under_diag,
        "over_diag": over_diag,
        "adjacent_errors": adjacent_errors,
        "multistep_errors": multistep_errors,
    }


# =========================================================
# CSV Saving
# =========================================================

def save_subset_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if output_path.exists():
            output_path.unlink()
        return

    extra_fields = [
        "error_direction",
        "severity_distance",
        "true_class_prob",
        "confidence_gap",
        "entropy",
    ]
    base_fields = [
        k for k in rows[0]
        if k not in {"probs", "severity_diff", *extra_fields}
    ]
    fieldnames = base_fields + extra_fields

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {k: row.get(k, "") for k in fieldnames}
            clean_row["true_class_prob"] = f"{row['true_class_prob']:.6f}"
            clean_row["confidence_gap"] = f"{row['confidence_gap']:.6f}"
            clean_row["entropy"] = f"{row['entropy']:.6f}"
            writer.writerow(clean_row)


# =========================================================
# HTML Report Generator
# =========================================================

def to_relative_web_path(path_str: str, base_dir: Path) -> str:
    if not path_str:
        return ""
    try:
        rel = os.path.relpath(Path(path_str), base_dir)
        return rel.replace("\\", "/")
    except ValueError:
        return Path(path_str).as_uri()


def generate_html_report(stats: dict, output_path: Path):
    errors_sorted = sorted(
        stats["errors"],
        key=lambda r: (r["confidence"], r["severity_distance"]),
        reverse=True,
    )

    cards_html = []
    for r in errors_sorted:
        is_high = r["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
        is_under = r["error_direction"] == "under_diagnosis"
        is_multi = r["severity_distance"] >= 2

        tags = ["all"]
        if is_high:
            tags.append("high_conf")
        if is_under:
            tags.append("under_diag")
        if is_multi:
            tags.append("multistep")

        badges = []
        if is_high:
            badges.append(
                '<span class="badge badge-danger">High Confidence</span>')
        if is_under:
            badges.append(
                '<span class="badge badge-warn">Under-Diagnosis</span>')
        else:
            badges.append(
                '<span class="badge badge-info">Over-Diagnosis</span>')
        if is_multi:
            badges.append(
                f'<span class="badge badge-critical">Distance: {r["severity_distance"]}</span>'
            )

        orig_src = to_relative_web_path(
            r.get("original_path", ""), output_path.parent)
        heat_src = to_relative_web_path(
            r.get("heatmap_path", ""), output_path.parent)
        over_src = to_relative_web_path(
            r.get("overlay_path", ""), output_path.parent)
        true_over_src = to_relative_web_path(
            r.get("true_class_overlay_path", ""), output_path.parent
        )

        prob_bars = []
        for cls_name in CLASS_ORDER:
            p = r["probs"].get(cls_name, 0.0)
            bar_cls = "bar-fill"
            if cls_name == r["predicted_class"]:
                bar_cls += " bar-pred"
            elif cls_name == r["true_class"]:
                bar_cls += " bar-true"
            prob_bars.append(
                f"""
                <div class="prob-row">
                    <span class="prob-label">{html.escape(cls_name)}</span>
                    <div class="bar-track"><div class="{bar_cls}" style="width:{p*100:.1f}%"></div></div>
                    <span class="prob-val">{p*100:.1f}%</span>
                </div>
                """
            )

        counterfactual_html = ""
        if true_over_src:
            counterfactual_html = f"""
            <figure>
                <img src="{html.escape(true_over_src)}" alt="True Class CAM" loading="lazy">
                <figcaption>True-Class CAM ({html.escape(r['true_class'])})</figcaption>
            </figure>
            """

        cards_html.append(
            f"""
            <article class="card" data-tags="{' '.join(tags)}">
                <header class="card-header">
                    <div>
                        <h3>{html.escape(r['image'])}</h3>
                        <p class="meta">
                            True: <strong>{html.escape(r['true_class'])}</strong> &rarr;
                            Predicted: <strong>{html.escape(r['predicted_class'])}</strong> |
                            Conf: <strong>{r['confidence']:.4f}</strong> |
                            True Prob: <strong>{r['true_class_prob']:.4f}</strong> |
                            Entropy: <strong>{r['entropy']:.3f}</strong>
                        </p>
                    </div>
                    <div class="badges">{''.join(badges)}</div>
                </header>
                <div class="images">
                    <figure>
                        <img src="{html.escape(orig_src)}" alt="Original" loading="lazy">
                        <figcaption>Original</figcaption>
                    </figure>
                    <figure>
                        <img src="{html.escape(heat_src)}" alt="Heatmap" loading="lazy">
                        <figcaption>Heatmap</figcaption>
                    </figure>
                    <figure>
                        <img src="{html.escape(over_src)}" alt="Predicted CAM Overlay" loading="lazy">
                        <figcaption>Predicted CAM ({html.escape(r['predicted_class'])})</figcaption>
                    </figure>
                    {counterfactual_html}
                </div>
                <div class="probs">
                    {''.join(prob_bars)}
                </div>
            </article>
            """
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DR Explainability Error Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1, h3 {{ margin: 0; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0; }}
  .kpi {{ background: #1e293b; padding: 16px; border-radius: 10px; border: 1px solid #334155; }}
  .kpi .val {{ font-size: 1.6rem; font-weight: 700; margin-top: 6px; }}
  .filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
  .filters button {{ background: #1e293b; color: #e2e8f0; border: 1px solid #475569; padding: 8px 14px; border-radius: 8px; cursor: pointer; }}
  .filters button.active {{ background: #3b82f6; border-color: #3b82f6; color: #fff; font-weight: 600; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 18px; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }}
  .meta {{ color: #94a3b8; font-size: 0.9rem; margin: 6px 0 0; }}
  .badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .badge {{ padding: 4px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }}
  .badge-danger {{ background: #991b1b; color: #fecaca; }}
  .badge-warn {{ background: #92400e; color: #fde68a; }}
  .badge-info {{ background: #1e40af; color: #bfdbfe; }}
  .badge-critical {{ background: #701a75; color: #f5d0fe; }}
  .images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 14px; }}
  figure {{ margin: 0; background: #0f172a; padding: 8px; border-radius: 8px; text-align: center; }}
  figure img {{ width: 100%; max-height: 240px; object-fit: contain; border-radius: 6px; }}
  figcaption {{ font-size: 0.8rem; color: #94a3b8; margin-top: 6px; }}
  .probs {{ display: grid; gap: 6px; }}
  .prob-row {{ display: grid; grid-template-columns: 130px 1fr 56px; align-items: center; gap: 10px; font-size: 0.85rem; }}
  .bar-track {{ background: #0f172a; height: 10px; border-radius: 6px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: #64748b; }}
  .bar-pred {{ background: #ef4444; }}
  .bar-true {{ background: #22c55e; }}
  .prob-val {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
  <h1>Diabetic Retinopathy — Grad-CAM Error Inspection Report</h1>
  <div class="summary">
    <div class="kpi"><div>Total Samples</div><div class="val">{stats['total']}</div></div>
    <div class="kpi"><div>Sample Accuracy</div><div class="val">{stats['accuracy']:.2%}</div></div>
    <div class="kpi"><div>Total Errors</div><div class="val">{stats['incorrect']}</div></div>
    <div class="kpi"><div>High-Conf Errors (&ge;{HIGH_CONFIDENCE_THRESHOLD:.2f})</div><div class="val">{len(stats['high_conf_errors'])}</div></div>
    <div class="kpi"><div>Under-Diagnosis</div><div class="val">{len(stats['under_diag'])}</div></div>
    <div class="kpi"><div>Multi-Step Errors (&ge;2)</div><div class="val">{len(stats['multistep_errors'])}</div></div>
  </div>
  <div class="filters">
    <button class="active" onclick="filterCards('all', this)">All Errors ({stats['incorrect']})</button>
    <button onclick="filterCards('high_conf', this)">High-Confidence ({len(stats['high_conf_errors'])})</button>
    <button onclick="filterCards('under_diag', this)">Under-Diagnosis ({len(stats['under_diag'])})</button>
    <button onclick="filterCards('multistep', this)">Multi-Step &ge;2 ({len(stats['multistep_errors'])})</button>
  </div>
  <section id="cards">
    {''.join(cards_html) if cards_html else '<p>No misclassified samples found.</p>'}
  </section>
  <script>
    function filterCards(tag, btn) {{
      document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.card').forEach(card => {{
        const tags = card.dataset.tags.split(' ');
        card.style.display = tags.includes(tag) ? 'block' : 'none';
      }});
    }}
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


# =========================================================
# Console Summary
# =========================================================

def print_summary(results: list[dict], stats: dict):
    print("\n" + "=" * 70)
    print("GRAD-CAM CLINICAL ERROR ANALYSIS")
    print("=" * 70)
    print(f"Total samples:            {stats['total']}")
    print(
        f"Correct / Incorrect:      {stats['correct']} / {stats['incorrect']} ({stats['accuracy']:.2%})")
    print(
        f"High-confidence errors:   {len(stats['high_conf_errors'])} (>= {HIGH_CONFIDENCE_THRESHOLD:.2f})")
    print(f"Low-confidence errors:    {len(stats['low_conf_errors'])}")
    print(
        f"Under-diagnosis errors:   {len(stats['under_diag'])} (clinically higher risk)")
    print(f"Over-diagnosis errors:    {len(stats['over_diag'])}")
    print(
        f"Adjacent vs Multi-step:   {len(stats['adjacent_errors'])} adjacent | {len(stats['multistep_errors'])} multi-step (>=2)")

    pairs = Counter(
        (r["true_class"], r["predicted_class"])
        for r in stats["errors"]
    )
    print("\nMost common confusion pairs:")
    for (t_cls, p_cls), count in pairs.most_common(10):
        print(f"  {t_cls} -> {p_cls}: {count}")

    if stats["high_conf_errors"]:
        print("\nTop High-Confidence Errors:")
        for idx, r in enumerate(
            sorted(stats["high_conf_errors"],
                   key=lambda x: x["confidence"], reverse=True),
            start=1,
        ):
            print(
                f"  {idx}. {r['image']} | {r['true_class']} -> {r['predicted_class']} | "
                f"Conf: {r['confidence']:.4f} | TrueProb: {r['true_class_prob']:.4f} | "
                f"{r['error_direction']} (dist={r['severity_distance']})"
            )


if __name__ == "__main__":
    results = prepare_results(load_results())
    stats = calculate_statistics(results)

    print_summary(results, stats)

    save_subset_csv(stats["errors"], ALL_ERRORS_PATH)
    save_subset_csv(stats["high_conf_errors"], HIGH_CONFIDENCE_ERRORS_PATH)
    save_subset_csv(stats["under_diag"], UNDER_DIAGNOSIS_ERRORS_PATH)
    generate_html_report(stats, HTML_REPORT_PATH)

    print("\n" + "=" * 70)
    print(f"HTML visual report:       {HTML_REPORT_PATH}")
    print(f"All errors CSV:           {ALL_ERRORS_PATH}")
    print(f"High-confidence CSV:      {HIGH_CONFIDENCE_ERRORS_PATH}")
    print(f"Under-diagnosis CSV:      {UNDER_DIAGNOSIS_ERRORS_PATH}")
    print("=" * 70)
