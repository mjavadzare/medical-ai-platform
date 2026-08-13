## Experiment 03 — ResNet50 Layer4 Fine-tuning with Focal Loss

**Date:** 2026-08-12

### Objective

Evaluate whether combining partial fine-tuning of a pretrained ResNet50 backbone with Focal Loss improves diabetic retinopathy classification performance, particularly for the minority classes.

This experiment follows the previous experiment using weighted CrossEntropyLoss and investigates whether Focal Loss can improve the model's ability to focus on difficult and misclassified samples.

### Model

- Architecture: ResNet50
- Pretrained weights: ImageNet
- Number of classes: 5
- Backbone: Partially unfrozen
- Trainable layer: Layer4 + final fully connected layer
- Frozen layers: Earlier ResNet50 layers

### Dataset

- Task: Diabetic Retinopathy Classification
- Number of classes: 5
- Classes:
  - 0 — No DR
  - 1 — Mild
  - 2 — Moderate
  - 3 — Severe
  - 4 — Proliferative DR
- Split strategy: Stratified
- Train/Test split: 80% / 20%
- Train/Validation split: 80% / 20%
- Random state: 10

### Preprocessing

- Images converted to RGB
- Images normalized to `[0, 1]` before calculating statistics
- Training mean and standard deviation calculated only from the training set
- Training augmentation:
  - Random horizontal flip
  - Random rotation
  - Color jitter
- Validation/test augmentation:
  - No augmentation

### Loss

- Loss function: Focal Loss
- Focal Loss was used instead of the previous weighted CrossEntropyLoss.
- The purpose was to reduce the contribution of easy samples and focus training more strongly on difficult and misclassified samples.

### Optimizer

- Optimizer: AdamW
- Initial learning rate: `1e-4`
- Weight decay: `1e-3`

### Scheduler

- Scheduler: ReduceLROnPlateau
- Mode: `min`
- Factor: `0.1`
- Patience: `2`
- Minimum learning rate: `1e-6`
- Monitored metric: Validation loss

### Training

- Epochs: 20
- Batch size: 32
- Device: CPU

### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
| ----: | ---------: | --------: | -------: | ------: | ---: |
| 1 | 0.5327 | 0.6846 | 0.3679 | 0.7611 | 1e-4 |
| 2 | 0.3214 | 0.7721 | 0.2939 | 0.7850 | 1e-4 |
| 3 | 0.2325 | 0.8173 | 0.2825 | 0.8020 | 1e-4 |
| 4 | 0.1987 | 0.8353 | 0.2797 | 0.7986 | 1e-4 |
| 5 | 0.1565 | 0.8638 | 0.2765 | 0.8072 | 1e-4 |
| 6 | 0.1252 | 0.8835 | 0.2777 | 0.8106 | 1e-4 |
| 7 | 0.1166 | 0.9014 | 0.2878 | 0.7935 | 1e-4 |
| 8 | 0.0851 | 0.9317 | 0.3231 | 0.7765 | 1e-5 |
| 9 | 0.0699 | 0.9381 | 0.3013 | 0.7935 | 1e-5 |
| 10 | 0.0591 | 0.9522 | 0.3032 | 0.7935 | 1e-5 |
| 11 | 0.0574 | 0.9543 | 0.3002 | 0.7969 | 1e-6 |
| 12 | 0.0517 | 0.9560 | 0.3019 | 0.7969 | 1e-6 |
| 13 | 0.0540 | 0.9501 | 0.3012 | 0.7952 | 1e-6 |
| 14 | 0.0500 | 0.9505 | 0.2987 | 0.8003 | 1e-6 |
| 15 | 0.0544 | 0.9501 | 0.3031 | 0.8038 | 1e-6 |
| 16 | 0.0534 | 0.9488 | 0.3066 | 0.7986 | 1e-6 |
| 17 | 0.0526 | 0.9518 | 0.3068 | 0.8020 | 1e-6 |
| 18 | 0.0477 | 0.9531 | 0.3082 | 0.7901 | 1e-6 |
| 19 | 0.0479 | 0.9522 | 0.3020 | 0.8003 | 1e-6 |
| 20 | 0.0487 | 0.9569 | 0.3176 | 0.7969 | 1e-6 |

### Best Validation Result

The best checkpoint was selected based on the lowest validation loss.

- **Best epoch:** 6
- **Best validation loss:** `0.2777` was not the best; the lowest validation loss was `0.2765` at epoch 5.
- **Best validation accuracy:** `81.06%` at epoch 6
- **Learning rate at best validation loss:** `1e-4`

Therefore, according to the checkpoint selection criterion based on validation loss:

- **Best checkpoint:** Epoch 5
- **Validation loss:** `0.2765`
- **Validation accuracy:** `80.72%`

The highest validation accuracy was obtained at epoch 6:

- **Validation accuracy:** `81.06%`
- **Validation loss:** `0.2777`

### Test Results

The final evaluation on the held-out test set produced:

- **Test Accuracy:** `80.35%`
- **Macro Precision:** `0.64`
- **Macro Recall:** `0.62`
- **Macro F1-score:** `0.62`
- **Weighted F1-score:** `0.80`

Per-class results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.96 | 0.97 | 0.97 | 361 |
| Mild | 0.59 | 0.72 | 0.65 | 74 |
| Moderate | 0.73 | 0.77 | 0.74 | 200 |
| Severe | 0.39 | 0.28 | 0.33 | 39 |
| Proliferative DR | 0.55 | 0.36 | 0.43 | 59 |

### Confusion Matrix

[[351   7   3   0   0]
 [  4  53  15   0   2]
 [ 10  23 153   8   6]
 [  0   1  18  11   9]
 [  1   6  22   9  21]]

### Comparison with Previous Experiment

The previous experiment used the same partially fine-tuned ResNet50 architecture with weighted CrossEntropyLoss.

| Metric | Layer4 + Weighted CE | Layer4 + Focal Loss | Change |
|---|---:|---:|---:|
| Test Accuracy | 78.72% | **80.35%** | **+1.63 pp** |
| Macro Precision | 0.64 | 0.64 | 0.00 |
| Macro Recall | 0.61 | **0.62** | +0.01 |
| Macro F1 | 0.62 | 0.62 | 0.00 |
| Weighted F1 | 0.78 | **0.80** | +0.02 |

Per-class changes:

| Class | F1 — Weighted CE | F1 — Focal Loss | Change |
|---|---:|---:|---:|
| No DR | 0.96 | **0.97** | +0.01 |
| Mild | 0.55 | **0.65** | **+0.10** |
| Moderate | 0.72 | **0.74** | +0.02 |
| Severe | **0.44** | 0.33 | **-0.11** |
| Proliferative DR | 0.45 | 0.43 | -0.02 |

### Observations

- Focal Loss improved overall test accuracy from `78.72%` to `80.35%`.
- Macro F1 remained approximately unchanged at `0.62`.
- Macro recall improved slightly from `0.61` to `0.62`.
- Weighted F1 improved from `0.78` to `0.80`.
- Mild-class performance improved substantially, with F1 increasing from `0.55` to `0.65`.
- Moderate-class performance also improved slightly, with F1 increasing from `0.72` to `0.74`.
- Severe-class performance decreased significantly, with F1 dropping from `0.44` to `0.33`.
- Proliferative DR performance decreased slightly, with F1 dropping from `0.45` to `0.43`.
- The model achieved very high performance on the majority class (`No DR`), with an F1-score of `0.97`.
- Training accuracy reached `95.69%`, while the best validation accuracy was `81.06%`, indicating substantial overfitting.
- Validation loss reached its minimum at epoch 5 (`0.2765`) and did not improve afterward.
- Reducing the learning rate after epoch 7 did not produce a meaningful improvement in validation loss.
- Focal Loss helped improve performance on the Mild and Moderate classes, but the improvement was not consistent across all minority classes.
- The Severe and Proliferative DR classes remain challenging, as reflected by their low recall and F1-scores.
- The confusion matrix shows considerable confusion between adjacent disease stages, particularly between Moderate, Severe, and Proliferative DR.

### Conclusion

Replacing weighted CrossEntropyLoss with Focal Loss produced a modest improvement in overall test performance, increasing test accuracy from `78.72%` to `80.35%`.

The most notable improvement was observed for the Mild class, whose F1-score increased from `0.55` to `0.65`. Moderate-class F1 also improved slightly from `0.72` to `0.74`.

However, performance on Severe and Proliferative DR remained weak. In particular, Severe-class F1 decreased from `0.44` to `0.33`.

Overall, Focal Loss improved the model's general performance but did not significantly improve macro F1 (`0.62` in both experiments). Therefore, Focal Loss alone does not appear sufficient to address the minority-class problem.

