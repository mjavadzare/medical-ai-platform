## Experiment 05: ResNet50 Layer3 + Layer4 Fine-Tuning with Focal Loss

**Date:** 2026-08-13

### Objective

Evaluate whether fine-tuning both `layer3` and `layer4` of a pretrained ResNet50 improves diabetic retinopathy classification compared with fine-tuning only `layer4`.

This experiment also changes the checkpoint selection criterion from validation loss to validation macro F1-score in order to better reflect performance across the imbalanced classes.

### Model

- Architecture: ResNet50
- Pretrained weights: ImageNet
- Number of classes: 5
- Backbone:
  - `layer1`: Frozen
  - `layer2`: Frozen
  - `layer3`: Trainable
  - `layer4`: Trainable
- Trainable layers:
  - `layer3`
  - `layer4`
  - Final fully connected classifier

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

The same preprocessing pipeline used in the previous experiment was retained.

- Images converted to RGB
- Images normalized to `[0, 1]` before calculating statistics
- Training mean and standard deviation calculated only from the training set
- Training augmentation:
  - Random horizontal flip
  - Random rotation
  - Color jitter
- Validation/test augmentation:
  - No augmentation

No cropping was introduced in this experiment.

### Loss

- Loss function: Focal Loss
- Gamma: `2.0`

Focal Loss was retained from the previous experiment to reduce the influence of easy examples and improve learning on difficult and minority-class samples.

### Optimizer

- Optimizer: AdamW
- Initial learning rate: `5e-5`
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
- Weighted sampler: No

### Checkpoint Selection

The checkpoint selection criterion was changed in this experiment.

Previously, the best model was selected based on the lowest validation loss.

Because the dataset is significantly class-imbalanced, validation loss and accuracy may not adequately represent performance on minority classes.

Therefore, the best checkpoint is now selected based on the highest validation **Macro F1-score**.

This gives equal importance to all five classes when determining the best model.

### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro F1 | LR |
|------:|-----------:|----------:|---------:|--------:|-------------:|---:|
| 1 | 0.6041 | 0.6581 | 0.4254 | 0.7406 | 0.3282 | 5e-5 |
| 2 | 0.3745 | 0.7367 | 0.3377 | 0.7765 | 0.4293 | 5e-5 |
| 3 | 0.2786 | 0.7900 | 0.2861 | 0.7884 | 0.5268 | 5e-5 |
| 4 | 0.2255 | 0.8233 | 0.2781 | 0.7901 | 0.5547 | 5e-5 |
| 5 | 0.1940 | 0.8412 | 0.2574 | 0.8225 | 0.6453 | 5e-5 |
| 6 | 0.1668 | 0.8562 | 0.2602 | 0.8140 | 0.6287 | 5e-5 |
| 7 | 0.1423 | 0.8660 | 0.2587 | 0.8311 | **0.6760** | 5e-5 |
| 8 | 0.1138 | 0.9031 | 0.2762 | 0.8208 | 0.6490 | 5e-6 |
| 9 | 0.0932 | 0.9125 | 0.2821 | 0.8191 | 0.6463 | 5e-6 |
| 10 | 0.0878 | 0.9253 | 0.2753 | 0.8140 | 0.6324 | 5e-6 |
| 11 | 0.0812 | 0.9279 | 0.2691 | 0.8191 | 0.6506 | 1e-6 |
| 12 | 0.0806 | 0.9245 | 0.2711 | 0.8191 | 0.6464 | 1e-6 |
| 13 | 0.0738 | 0.9356 | 0.2721 | 0.8191 | 0.6569 | 1e-6 |
| 14 | 0.0708 | 0.9330 | 0.2723 | 0.8123 | 0.6381 | 1e-6 |
| 15 | 0.0833 | 0.9240 | 0.2650 | 0.8208 | 0.6639 | 1e-6 |
| 16 | 0.0708 | 0.9347 | 0.2725 | 0.8140 | 0.6400 | 1e-6 |
| 17 | 0.0773 | 0.9364 | 0.2820 | 0.8174 | 0.6441 | 1e-6 |
| 18 | 0.0744 | 0.9364 | 0.2751 | 0.8174 | 0.6497 | 1e-6 |
| 19 | 0.0759 | 0.9292 | 0.2758 | 0.8123 | 0.6454 | 1e-6 |
| 20 | 0.0777 | 0.9283 | 0.2735 | 0.8140 | 0.6505 | 1e-6 |

### Best Validation Result

The best checkpoint was selected based on the highest validation Macro F1-score.

- **Best epoch:** 7
- **Best validation Macro F1:** `0.6760`
- **Validation accuracy:** `0.8311`
- **Validation loss:** `0.2587`
- **Learning rate:** `5e-5`

The best validation Macro F1 occurred at epoch 7.

### Test Results

The selected best checkpoint was evaluated on the held-out test set.

- **Test Accuracy:** `0.8090`
- **Test Macro F1:** `0.64`

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.96 | 0.98 | 0.97 | 361 |
| Mild | 0.64 | 0.58 | 0.61 | 74 |
| Moderate | 0.71 | 0.79 | 0.75 | 200 |
| Severe | 0.39 | 0.36 | 0.37 | 39 |
| Proliferative DR | 0.57 | 0.41 | 0.48 | 59 |

- Macro average:
  - Precision: `0.66`
  - Recall: `0.62`
  - F1: `0.64`
- Weighted average:
  - Precision: `0.80`
  - Recall: `0.81`
  - F1: `0.80`

### Confusion Matrix


[[355   4   2   0   0]
 [  4  43  26   0   1]
 [  8  17 157   9   9]
 [  0   0  17  14   8]
 [  1   3  18  13  24]]


### Observations

- Fine-tuning both `layer3` and `layer4` improved validation performance compared with the previous `layer4`-only configuration.
- The best validation Macro F1 increased to **0.6760**, compared with **0.6220** for the previous `ResNet50_layer4_ft_focal_sampler` experiment.
- Validation accuracy reached **0.8311** at its best point.
- The model achieved a **test accuracy of 0.8090** and a **test Macro F1 of 0.64**.
- `No DR` remained the strongest class with an F1-score of **0.97**.
- `Moderate` also performed relatively well with an F1-score of **0.75**.
- Performance on the minority classes remained weaker, especially `Severe` (F1: **0.37**) and `Proliferative DR` (F1: **0.48**).
- The training loss continued decreasing while validation Macro F1 peaked around epoch 7, indicating that further training mainly increased overfitting rather than improving generalization.
- The learning rate was reduced from `5e-5` to `5e-6` after epoch 7 and eventually to `1e-6`.
- The criterion for selecting the best checkpoint was changed from **validation loss** to **validation Macro F1**, since the dataset is highly imbalanced and Macro F1 better reflects performance across all classes.

### Result

**Model:** `ResNet50_layer3_layer4_ft_focal`

**Best validation performance:**
- Val Accuracy: **0.8311**
- Val Macro F1: **0.6760**

**Test performance:**
- Accuracy: **0.8090**
- Macro F1: **0.64**
- Weighted F1: **0.80**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| No DR | 0.96 | 0.98 | 0.97 |
| Mild | 0.64 | 0.58 | 0.61 |
| Moderate | 0.71 | 0.79 | 0.75 |
| Severe | 0.39 | 0.36 | 0.37 |
| Proliferative DR | 0.57 | 0.41 | 0.48 |

The model showed an improvement in validation Macro F1 after fine-tuning both `layer3` and `layer4`, but minority-class performance remained the main limitation.

### Conclusion

Fine-tuning `layer3` in addition to `layer4` produced a meaningful improvement in validation Macro F1. However, the improvement did not translate into a higher test accuracy compared with the previous experiment.

The main remaining issue is the poor performance on the minority classes, particularly `Severe` and `Proliferative DR`.
