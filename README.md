# Medical AI Platform — Diabetic Retinopathy Screening & Explainability

[![CI](https://github.com/mjavadzare/medical-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/mjavadzare/medical-ai-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-285A48?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50%20%7C%20Focal%20Loss-408A71?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving%20%26%20Audit%20Log-091413?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-WSL2%20%26%20Compose-285A48?logo=docker&logoColor=white)

A modular medical artificial intelligence platform for **5-class Diabetic Retinopathy (DR) grading**, **post-hoc probability calibration (Temperature Scaling)**, **Grad-CAM++ visual explainability**, and **clinical triage support**, built with PyTorch, FastAPI, SQLAlchemy, and Docker.

> **Research & Clinical Disclaimer:** This system is designed for research, prototyping, and decision-support workflows. It is not a certified medical device and must not replace evaluation by a qualified ophthalmologist.

---

## Key Capabilities

* **5-Class Retinopathy Grading (`0`–`4`):** Classifies fundus images into `No_DR`, `Mild`, `Moderate`, `Severe`, and `Proliferative_DR` using fine-tuned `ResNet50` (`ResNet50_layer3_layer4_ft_bn_frozen_focal`) trained with Focal Loss.
* **Post-Hoc Temperature Scaling & Reliability Diagrams:** Fits a scalar temperature ($T = 0.9196$ via L-BFGS minimizing NLL on the validation split) to improve probability calibration while preserving classification accuracy, accompanied by before/after reliability diagrams and ECE/MCE/NLL/Brier metrics.
* **Explainable AI (`Grad-CAM++`):** Generates class-discriminative saliency overlays (`layer4` by default, configurable to `layer3`) encoded as Base64 PNGs for visual inspection.
* **Clinical Triage & Uncertainty Signals:** Flags referable DR cases (`Moderate`, `Severe`, `Proliferative_DR`) as well as low-confidence or high-entropy predictions for specialist review.
* **Production-Ready API, Audit Logging & CI:** Exposes REST endpoints via FastAPI, persists inference history with SQLAlchemy, containerizes cleanly with Docker Compose, and runs `ruff` + `pytest` on every push via GitHub Actions.

---

## Probability Calibration (Temperature Scaling)

Focal Loss improves minority/referable class learning, but raw softmax probabilities can be slightly underconfident. Post-hoc Temperature Scaling ($T = 0.9196$, fitted strictly on the stratified validation split using L-BFGS minimizing NLL) refines probability estimates without changing `argmax` predictions or accuracy:

| Split | Samples | ECE (Before → After) | NLL (Before → After) | Brier (Before → After) | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation** | 586 | 0.0412 → **0.0333** (-19.10%) | 0.4754 → **0.4738** | 0.2411 → **0.2393** | 83.45% |
| **Test (Held-out)** | 733 | 0.0477 → **0.0435** (-8.88%) | 0.4468 → **0.4415** | 0.2337 → **0.2308** | 84.17% |

![Test Reliability Diagram](reports/calibration/reliability_diagram_test.png)

---

## Repository Structure

```text
medical-ai-platform/
├── app/                                  # FastAPI application, settings, schemas, DB & routers
│   ├── api/                              # Health and DR inference/explain/history routes
│   └── core/                             # Configuration (including CALIBRATION_TEMPERATURE)
├── modules/
│   └── diabetic_retinopathy/
│       ├── artifacts/checkpoints/        # Trained model checkpoints (.pth)
│       ├── datasets/                     # Dataset factory, PyTorch Dataset & DataLoaders
│       ├── explainability/               # Grad-CAM batch runner & analysis tools
│       ├── inference/                    # DRPredictor & preprocessing/overlay utilities
│       ├── models/                       # ResNet50, DenseNet121, and EfficientNet-B0 builders
│       ├── training/                     # Training engine, Focal Loss & trainer loop
│       └── calibrate.py                  # CLI script for validation fitting & test calibration report
├── shared/
│   ├── calibration/                      # TemperatureScaler, ECE/NLL/Brier & reliability plots
│   ├── datasets/                         # Stratified train/val/test split utilities
│   ├── explainability/                   # Grad-CAM & Grad-CAM++ implementations
│   ├── metrics/                          # Classification metrics & evaluation helpers
│   └── preprocessing/                    # Dataset statistics & image transforms
├── mlops/
│   ├── Dockerfile                        # Container image definition
│   └── docker-compose.yml                # Service orchestration & environment configuration
├── tests/                                # API and calibration unit tests (pytest)
├── reports/calibration/                  # Generated calibration_report.json & reliability PNGs
├── pyproject.toml                        # Package metadata, dependencies & Ruff configuration
└── .github/workflows/ci.yml              # Automated linting (ruff) and test suite (pytest)
```

---

## Quick Start

### 1. Local Environment Setup (Windows / Linux)

```bash
git clone https://github.com/mjavadzare/medical-ai-platform.git
cd medical-ai-platform
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -e ".[dev]"
```

### 2. Run Calibration & Generate Reliability Diagrams

```bash
python -m modules.diabetic_retinopathy.calibrate \
  --csv-path data/processed/diabetic_retinopathy/image_metadata.csv \
  --checkpoint modules/diabetic_retinopathy/artifacts/checkpoints/ResNet50_layer3_layer4_ft_bn_frozen_focal/ResNet50_layer3_layer4_ft_bn_frozen_focal_best_model.pth
```

### 3. Run FastAPI Service Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs` and health status at `http://127.0.0.1:8000/health`.

### 4. Run with Docker Compose

```bash
docker compose -f mlops/docker-compose.yml up --build
```

---

## Quality Assurance & CI

Run linting and unit tests locally before pushing changes:

```bash
ruff check .
pytest -q
```

---

## Author & Contact

**Mohammad Javad Zare** — AI Researcher & AI Software Engineer (Computer Vision, Medical Imaging, VLMs, Multimodal & Agentic AI)

* **Website:** [mjavadzare.github.io](https://mjavadzare.github.io)
* **GitHub:** [github.com/mjavadzare](https://github.com/mjavadzare)
* **LinkedIn:** [linkedin.com/in/mohammad-javad-zare](https://www.linkedin.com/in/mohammad-javad-zare/)
* **Email:** `mjz589.2018@gmail.com`