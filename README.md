# Medical AI Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-3.95x%20CPU%20Speedup-005CED?logo=onnx&logoColor=white)](#onnx-runtime-optimization--cpu-benchmark)
[![Dataset: Kaggle](https://img.shields.io/badge/Dataset-Kaggle%20DR%20224x224-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM%20%26%20Grad--CAM%2B%2B-0077B6)](#explainability--clinical-error-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-023E8A)](LICENSE)

A modular, experiment-driven medical imaging platform focused on **5-class Diabetic Retinopathy (DR) severity grading** using deep transfer learning, class-imbalance mitigation, post-hoc probability calibration (**Temperature Scaling**), **Grad-CAM / Grad-CAM++** visual explainability, clinical triage support, **ONNX Runtime** CPU acceleration, and **FastAPI / Streamlit / Docker / CI** deployment.

> **Disclaimer:** This project is developed strictly for research and educational purposes. The trained models are not cleared for clinical diagnosis and must not replace evaluation by qualified medical professionals.

---

## Key Highlights

-   **Modular Platform Architecture:** Clean separation between reusable infrastructure (`shared/`) and disease-specific modules (`modules/diabetic_retinopathy/`), paired with a decoupled **FastAPI** serving layer (`app/`), **Streamlit** clinical dashboard (`ui/`), and SQLAlchemy audit logging.
-   **Controlled Experimentation (9 Documented Runs):** Incremental ablations across backbones (`EfficientNet-B0`, `ResNet50`, `DenseNet121`), layer-wise fine-tuning depth, BatchNorm freezing, Focal Loss ($\gamma = 2.0$), and inverse-frequency `WeightedRandomSampler` — all logged in [`docs/`](docs/).
-   **Strong Clinical Screening Performance:** Best model (`ResNet50_layer3_layer4_ft_bn_frozen_focal`) achieves **84.17% Test Accuracy**, **0.70 Macro F1**, **0.898 Quadratic Weighted Kappa (QWK)**, and **93.6% Sensitivity / 94.7% Specificity** on Referable DR (`Moderate` or higher).
-   **Post-Hoc Calibration & Reliability Diagrams:** Fits a scalar temperature ($T = 0.9196$ via L-BFGS minimizing NLL on the validation split) to improve probability calibration while preserving classification accuracy.
-   **ONNX Runtime CPU Acceleration:** Exports the calibrated `ResNet50` model to ONNX (opset 17) with verified numerical parity (max probability diff $2.1 \times 10^{-7}$), cutting CPU inference latency from **132.69 ms** to **33.57 ms** (**3.95x speedup**) and p95 latency from **229.10 ms** to **45.37 ms**.
-   **Counterfactual Explainability & Error Auditing:** Reusable `GradCAM` and `GradCAMPlusPlus` engines that generate both predicted-class and true-class (counterfactual) overlays on misclassified scans, plus an automated HTML inspection report isolating high-confidence and under-diagnosis errors.
-   **Clinical Screening UI & Hardened CI/CD:** Decoupled hospital-themed Streamlit dashboard (`dr-ui`) communicating with FastAPI (`dr-api`) over Docker Compose, backed by GitHub Actions CI running `ruff`, 11 `pytest` tests (including ONNX parity), and automated Docker Buildx checks for both containers.

---

## System Architecture

```mermaid
flowchart LR
    A[Retinal Fundus Images<br/>224x224 RGB] --> B[Dataset & Stratified Split]
    B --> C[PyTorch DataLoader]
    C --> D[Pretrained CNN Backbone<br/>ResNet50 / DenseNet121 / EfficientNet]
    D --> E[Layer-Wise Fine-Tuning<br/>BN Frozen + Focal Loss]
    E --> F[Checkpointing<br/>Best Macro-F1 & Resume State]
    F --> G[Test Evaluation &<br/>Temperature Scaling]
    F --> H[Shared XAI Engine<br/>Grad-CAM / Grad-CAM++]
    F --> O[ONNX Export &<br/>Parity Benchmark]
    H --> I[Clinical Error Analyzer<br/>CSV Audits & HTML Visual Report]
    G --> J[FastAPI Serving<br/>ONNX Predict + PyTorch XAI<br/>& SQLAlchemy Audit Log]
    O --> J
    H --> J
    J <--> K[Clinical Dashboard<br/>Streamlit UI]
```

---

## Dataset

This project uses the **[Diabetic Retinopathy 224x224 (2019 Data)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)** dataset from Kaggle (a pre-resized $224 \times 224$ version of the APTOS 2019 Blindness Detection retinal fundus imaging dataset).

-   **Source:** [Kaggle — Diabetic Retinopathy 224x224 (2019 Data)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)
-   **Processed Metadata:** `data/processed/diabetic_retinopathy/image_metadata.csv`
-   **Additional Notes:** [`docs/dataset.md`](docs/dataset.md)
-   **Split Strategy:** Stratified Train / Validation / Test split preserving the natural class imbalance across all five DR severity grades.

### Class Distribution (Test Set: 733 Retinal Scans)

The strong class imbalance is a primary challenge addressed throughout the experiments (via Focal Loss, inverse-frequency sampling, and Macro F1 model selection).

| Label | Class Name           | Clinical Severity                           | Test Support |  Share   |
| :---: | :------------------- | :------------------------------------------ | :----------: | :------: |
|  `0`  | **No DR**            | No diabetic retinopathy                     |     361      |  49.2%   |
|  `1`  | **Mild**             | Mild non-proliferative DR                   |      74      |  10.1%   |
|  `2`  | **Moderate**         | Moderate non-proliferative DR _(Referable)_ |     200      |  27.3%   |
|  `3`  | **Severe**           | Severe non-proliferative DR _(Referable)_   |      39      |   5.3%   |
|  `4`  | **Proliferative DR** | Proliferative DR _(Referable)_              |      59      |   8.0%   |
|   —   | **Total**            | —                                           |   **733**    | **100%** |

---

## Benchmark Results

Model selection prioritizes **Validation Macro F1** alongside overall accuracy to prevent majority-class collapse.

| Experiment        | Model Configuration                                 |  Test Acc  | Macro F1 |  No DR   |   Mild   | Moderate |  Severe  | Prolif. DR |              Report              |
| :---------------- | :-------------------------------------------------- | :--------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------------------------------: |
| Exp 06            | `EfficientNet-B0`                                   |   71.90%   |   0.40   |    —     |    —     |    —     |    —     |     —      | [Details](docs/experiment-06.md) |
| **Exp 07 (Best)** | **`ResNet50_layer3_layer4_ft_bn_frozen_focal`**     | **84.17%** | **0.70** | **0.98** | **0.66** |   0.79   |   0.47   |  **0.59**  | [Details](docs/experiment-07.md) |
| Exp 08            | `ResNet50_layer3_layer4_ft_bn_frozen_focal_sampler` |   84.04%   |   0.70   |   0.98   |   0.63   | **0.80** | **0.48** |    0.59    | [Details](docs/experiment-08.md) |
| Exp 09            | `DenseNet121_denseblock4_ft_focal`                  |   75.03%   |   0.58   |   0.97   |   0.57   |   0.63   |   0.32   |    0.41    | [Details](docs/experiment-09.md) |

_(Earlier baseline ablations for ResNet50 classifier-only and `layer4` fine-tuning are documented in [`docs/experiment-01.md`](docs/experiment-01.md) through [`docs/experiment-05.md`](docs/experiment-05.md).)_

### Clinical Screening Metrics (Best Model)

Derived from the 733-sample test confusion matrix:

| Clinical Metric                               |   Score   | Notes                                                           |
| :-------------------------------------------- | :-------: | :-------------------------------------------------------------- |
| **Quadratic Weighted Kappa (QWK)**            | **0.898** | Standard ordinal agreement metric for DR grading                |
| **Referable DR Sensitivity** ($\ge$ Moderate) | **93.6%** | 279 / 298 referable cases correctly flagged                     |
| **Referable DR Specificity** (No DR / Mild)   | **94.7%** | 412 / 435 non-referable cases correctly cleared                 |
| **Adjacent-Class Error Share**                | **73.3%** | 85 / 116 total errors occur between neighboring severity grades |

<details>
<summary><strong>View Best Model Confusion Matrix (Test Set)</strong></summary>

```text
True \ Pred       No DR   Mild   Moderate   Severe   Prolif. DR
No DR               353      7          1        0            0
Mild                  4     48         21        0            1
Moderate              1     15        170        9            5
Severe                0      0         17       17            5
Proliferative DR      1      2         20        7           29
```

![Best Model Confusion Matrix](modules/diabetic_retinopathy/artifacts/plots/ResNet50_layer3_layer4_ft_bn_frozen_focal/ResNet50_layer3_layer4_ft_bn_frozen_focal_best_confusion_matrix.png)

</details>

---

## Probability Calibration (Temperature Scaling)

Models trained with Focal Loss can exhibit slight underconfidence on softmax outputs. Post-hoc Temperature Scaling ($T = 0.9196$, fitted strictly on the stratified validation split using L-BFGS minimizing NLL) sharpens probability estimates while leaving `argmax` predictions and accuracy unchanged:

| Split               | Samples |     ECE (Before $\to$ After)      | NLL (Before $\to$ After) | Brier (Before $\to$ After) | Accuracy |
| :------------------ | :-----: | :-------------------------------: | :----------------------: | :------------------------: | :------: |
| **Validation**      |   586   | 0.0412 $\to$ **0.0333** (-19.10%) | 0.4754 $\to$ **0.4738**  |  0.2411 $\to$ **0.2393**   |  83.45%  |
| **Test (Held-out)** |   733   | 0.0477 $\to$ **0.0435** (-8.88%)  | 0.4468 $\to$ **0.4415**  |  0.2337 $\to$ **0.2308**   |  84.17%  |

The calibrated temperature (`CALIBRATION_TEMPERATURE=0.9196`) is wired into `app/core/config.py`, `DRPredictor`, and `mlops/docker-compose.yml`, with saved artifacts under `reports/calibration/` (`calibration_report.json`, `reliability_diagram_val.png`, and `reliability_diagram_test.png`).

![Test Reliability Diagram](reports/calibration/reliability_diagram_test.png)

---

## ONNX Runtime Optimization & CPU Benchmark

To optimize CPU-only clinical serving without sacrificing explainability, `DRPredictor` uses a **hybrid inference backend**:

-   **Fast Prediction Path (`/api/v1/dr/predict`):** Runs on **ONNX Runtime** (`ORT_ENABLE_ALL`, opset 17) when an exported `.onnx` model is available, applying post-hoc Temperature Scaling ($T = 0.9196$) dynamically to raw logits so calibration remains runtime-configurable without re-exporting. Automatically falls back to PyTorch if no `.onnx` file is present.
-   **Explainability Path (`/api/v1/dr/explain`):** Retains the PyTorch backbone for gradient-based **Grad-CAM++** saliency generation.

### CPU Latency Benchmark (`batch_size = 1`, $224 \times 224$, 10 warmup / 50 timed runs)

| Runtime Backend (CPU)  | Mean (ms) | Median (ms) | p95 (ms)  | Min (ms)  | Throughput (img/s) |  Speedup  |
| :--------------------- | :-------: | :---------: | :-------: | :-------: | :----------------: | :-------: |
| **PyTorch (CPU)**      |  132.69   |   111.66    |  229.10   |   96.04   |        7.54        |   1.00x   |
| **ONNX Runtime (CPU)** | **33.57** |  **34.57**  | **45.37** | **22.60** |     **29.79**      | **3.95x** |

-   **Numerical Parity Verification (`batch_size` $\in \{1, 4\}$):** Passed with `argmax_match = True`, max absolute logit diff $= 6.9 \times 10^{-7}$, and max calibrated probability diff $= 2.1 \times 10^{-7}$ (tolerance $10^{-4}$).
-   **Benchmark Artifact:** [`reports/onnx/onnx_benchmark.json`](reports/onnx/onnx_benchmark.json)

---

## Explainability & Clinical Error Analysis

Beyond aggregate accuracy, the platform runs a dedicated **Grad-CAM / Grad-CAM++** audit over a balanced 50-image diagnostic sample set (`10 images per class`) to investigate _where_ the model focuses and _why_ misclassifications occur.

### Audit Findings (50 Balanced Diagnostic Samples)

-   **Overall Sample Accuracy:** `35 / 50 (70.0%)`
-   **Dominant Failure Mode — Under-Diagnosis:** `11 / 15` errors (`73.3%`) under-predict disease severity, primarily `Severe` $\to$ `Moderate` (4 cases) and `Proliferative DR` $\to$ `Moderate` (3 cases).
-   **Confidence Separation:** `14 / 15` errors occur below `0.80` confidence, supporting a confidence/entropy-based **specialist review triage threshold** during deployment.
-   **High-Confidence Error Case Study (`sample_034.png`):** True label `Proliferative DR`, predicted `Severe` with **96.01% confidence** (`P(Proliferative DR) = 2.70%`). Comparing the predicted-class CAM against the counterfactual true-class CAM highlights the exact retinal regions driving the misclassification:

|                 Original Fundus                  |             Predicted CAM (`Severe`, 96.0%)              |          Counterfactual True-Class CAM (`Proliferative DR`)          |
| :----------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------------------: |
| ![Original](docs/assets/sample_034_original.png) | ![Predicted Overlay](docs/assets/sample_034_overlay.png) | ![True-Class Overlay](docs/assets/sample_034_true_class_overlay.png) |

Running `analyze_gradcam.py` exports structured CSV breakdowns (`all_errors.csv`, `high_confidence_errors.csv`, `under_diagnosis_errors.csv`) along with an offline interactive HTML inspection dashboard (`error_report.html`).

---

## Clinical Screening Dashboard (Streamlit)

A decoupled clinical interface (`ui/streamlit_app.py`) styled with a clinical blue palette connects directly to the FastAPI service over HTTP without duplicating model weights or PyTorch dependencies:

-   **Side-by-Side Visual Inspection:** Displays the uploaded (or local sample) retinal fundus scan alongside its **Grad-CAM++** saliency overlay (`layer4`).
-   **Calibrated Confidence & Uncertainty:** Visualizes 5-class calibrated probabilities ($T = 0.9196$), predictive entropy, and inference latency.
-   **Clinical Triage Alerts:** Highlights `NEEDS_SPECIALIST_REVIEW` vs. `ROUTINE_SCREENING` decisions with explicit review reasons.
-   **Audit History Tab:** Queries `/api/v1/dr/history` from the SQLAlchemy backend with live filtering by triage status.

![Clinical Screening Dashboard](docs/assets/dashboard_preview.jpg)

---

## Project Structure

```text
medical-ai-platform/
├── .streamlit/
│   └── config.toml                       # Clinical blue theme & Streamlit server settings
├── app/                                  # FastAPI serving layer (API, schemas, services, DB)
├── data/
│   ├── raw/                              # Downloaded Kaggle dataset (gitignored)
│   ├── processed/diabetic_retinopathy/   # Stratified metadata CSV
│   └── samples/diabetic_retinopathy/     # Balanced diagnostic samples for XAI audits
├── docs/                                 # Dataset notes, experiment-01..09 logs & visual assets
├── mlops/
│   ├── Dockerfile                        # FastAPI CPU serving image (PyTorch + ONNX Runtime)
│   ├── Dockerfile.ui                     # Lightweight Streamlit UI image (no PyTorch)
│   └── docker-compose.yml                # Multi-service orchestration (dr-api + dr-ui)
├── modules/
│   └── diabetic_retinopathy/
│       ├── artifacts/                    # Checkpoints & ONNX weights (gitignored), metrics, plots
│       ├── configs/                      # YAML configs (data, model, training)
│       ├── datasets/                     # Dataset & DataLoader factories
│       ├── evaluation/                   # CLI test-set evaluation script
│       ├── explainability/               # Grad-CAM++ runner & clinical error analyzer
│       ├── inference/                    # Hybrid ONNX/PyTorch predictor & preprocessing
│       ├── models/                       # ResNet50, DenseNet121, EfficientNet-B0
│       ├── training/                     # Trainer, engine, Focal Loss, checkpoint resuming
│       ├── transforms/                   # Training & evaluation image pipelines
│       ├── calibrate.py                  # CLI calibration script (Validation fit & Test report)
│       └── export_onnx.py                # CLI ONNX exporter, parity verifier & CPU benchmark
├── notebooks/                            # Exploratory data analysis & training notebooks
├── reports/
│   ├── calibration/                      # Calibration JSON & before/after reliability diagrams
│   └── onnx/                             # ONNX parity & CPU latency benchmark JSON
├── shared/                               # Reusable platform components
│   ├── calibration/                      # TemperatureScaler, ECE/NLL/Brier & reliability plots
│   ├── datasets/                         # Splitting utilities
│   ├── evaluation/                       # Generic classification evaluator
│   ├── explainability/                   # Reusable GradCAM, GradCAM++, overlay renderer
│   ├── metrics/                          # Classification metrics
│   └── visualization/                    # Confusion matrix plotting
├── tests/                                # Automated test suite (API, calibration, XAI, ONNX parity)
├── ui/
│   └── streamlit_app.py                  # Decoupled Streamlit clinical dashboard
├── .github/workflows/ci.yml              # GitHub Actions CI (ruff + pytest + Docker Buildx)
├── pyproject.toml
├── requirements-train.txt
├── requirements-serve.txt
└── requirements-ui.txt
```

---

## Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/mjavadzare/medical-ai-platform.git
cd medical-ai-platform

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

pip install -r requirements-train.txt
pip install -e . --no-build-isolation
```

### 2. Prepare the Dataset

Download the **[Diabetic Retinopathy 224x224 (2019 Data)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)** archive from Kaggle and extract it under `data/raw/`. Metadata splits are tracked in `data/processed/diabetic_retinopathy/image_metadata.csv`.

### 3. Training

```bash
python -u -m modules.diabetic_retinopathy.training.train
```

Training automatically manages checkpoints:

-   `<ModelName>_best_model.pth` — Saved at the highest Validation Macro F1.
-   `<ModelName>_last_checkpoint.pth` — Stores full epoch, model, optimizer, and scheduler state for seamless resuming.

### 4. Evaluation, Calibration & ONNX Export

```bash
# Test-set evaluation
python -u -m modules.diabetic_retinopathy.evaluation.evaluate \
    --model-name ResNet50_layer3_layer4_ft_bn_frozen_focal \
    --checkpoint-type best

# Fit Temperature Scaling on validation split & evaluate on held-out test split
python -m modules.diabetic_retinopathy.calibrate \
    --csv-path data/processed/diabetic_retinopathy/image_metadata.csv \
    --checkpoint modules/diabetic_retinopathy/artifacts/checkpoints/ResNet50_layer3_layer4_ft_bn_frozen_focal/ResNet50_layer3_layer4_ft_bn_frozen_focal_best_model.pth

# Export ResNet50 to ONNX (opset 17), verify parity & benchmark CPU latency
python -m modules.diabetic_retinopathy.export_onnx
```

### 5. Grad-CAM++ Explainability & Error Analysis

```bash
# Generate Grad-CAM++ heatmaps, overlays, and counterfactual maps for misclassifications
python -u -m modules.diabetic_retinopathy.explainability.gradcam_runner \
    --method gradcam++ \
    --layer layer4

# Run clinical error audit & build interactive HTML inspection report
python -u -m modules.diabetic_retinopathy.explainability.analyze_gradcam
```

### 6. Serving & Clinical Dashboard (FastAPI + Streamlit + Docker)

**Option A — Multi-Service Docker Compose (Recommended):**

```bash
docker compose -f mlops/docker-compose.yml up --build -d
```

Once running, both services are accessible locally:

-   **FastAPI Interactive Docs:** `http://127.0.0.1:8000/docs`
-   **Streamlit Clinical Dashboard:** `http://127.0.0.1:8502`

**Option B — Local Development:**

```bash
# Terminal 1: FastAPI backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit clinical dashboard
pip install -r requirements-ui.txt
streamlit run ui/streamlit_app.py --server.port 8502
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
