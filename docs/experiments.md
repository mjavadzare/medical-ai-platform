# Experiments

This file records the experiments performed during model development.

The purpose is to keep track of training configurations, model changes, evaluation results, and observations.

---

## Experiment 01 — ResNet50 with High Learning Rate

**Date:** 2026-08-07

### Objective

Evaluate the effect of using a relatively high initial learning rate when training a pretrained ResNet50 for diabetic retinopathy classification.

### Model

* Architecture: ResNet50
* Pretrained weights: ImageNet
* Number of classes: 5
* Backbone: Frozen
* Trainable layer: Final fully connected layer

### Dataset

* Task: Diabetic Retinopathy Classification
* Number of classes: 5
* Classes:

  * 0 — No DR
  * 1 — Mild
  * 2 — Moderate
  * 3 — Severe
  * 4 — Proliferative DR
* Split strategy: Stratified
* Train/Test split: 80% / 20%
* Train/Validation split: 80% / 20%
* Random state: 10

### Preprocessing

* Images converted to RGB
* Images normalized to `[0, 1]` before calculating statistics
* Training mean and standard deviation calculated only from the training set
* Training augmentation:

  * Random horizontal flip
  * Random rotation
  * Color jitter
* Validation/test augmentation:

  * No augmentation

### Loss

* Loss function: Weighted CrossEntropyLoss
* Class weights: Yes
* Class weights calculated from the training set

### Optimizer

* Optimizer: AdamW
* Initial learning rate: `0.01`
* Weight decay: `0.001`

### Scheduler

* Scheduler: ReduceLROnPlateau
* Mode: `min`
* Factor: `0.1`
* Patience: `2`
* Minimum learning rate: `1e-6`
* Monitored metric: Validation loss

### Training

* Epochs: 20
* Batch size: 32

### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |   LR |
| ----: | ---------: | --------: | -------: | ------: | ---: |
|     1 |     1.3379 |    0.6065 |   1.1403 |  0.7201 | 1e-2 |
|     2 |     1.0610 |    0.6927 |   1.5766 |  0.6075 | 1e-2 |
|     3 |     0.9466 |    0.7354 |   1.8179 |  0.6997 | 1e-2 |
|     4 |     0.9616 |    0.7264 |   1.2972 |  0.7577 | 1e-3 |
|     5 |     0.7537 |    0.7904 |   1.1533 |  0.7577 | 1e-3 |
|     6 |     0.7340 |    0.7900 |   1.1333 |  0.7577 | 1e-3 |
|     7 |     0.7142 |    0.7879 |   1.1581 |  0.7628 | 1e-3 |
|     8 |     0.7000 |    0.7913 |   1.0701 |  0.7270 | 1e-3 |
|     9 |     0.6835 |    0.8032 |   1.0558 |  0.7423 | 1e-3 |
|    10 |     0.7168 |    0.7781 |   1.0549 |  0.7526 | 1e-3 |
|    11 |     0.6833 |    0.7934 |   1.0925 |  0.7577 | 1e-3 |
|    12 |     0.6662 |    0.8105 |   1.0914 |  0.7270 | 1e-3 |
|    13 |     0.6553 |    0.8050 |   1.0583 |  0.7287 | 1e-4 |
|    14 |     0.6619 |    0.7960 |   1.0818 |  0.7526 | 1e-4 |
|    15 |     0.6490 |    0.8092 |   1.0603 |  0.7543 | 1e-4 |
|    16 |     0.6503 |    0.8084 |   1.0591 |  0.7440 | 1e-5 |
|    17 |     0.6372 |    0.7977 |   1.0721 |  0.7440 | 1e-5 |
|    18 |     0.6611 |    0.8045 |   1.0654 |  0.7491 | 1e-5 |
|    19 |     0.6553 |    0.8126 |   1.0800 |  0.7509 | 1e-6 |
|    20 |     0.6405 |    0.8092 |   1.0704 |  0.7491 | 1e-6 |

### Best Validation Result

The best checkpoint was selected based on the lowest validation loss.

* **Best epoch:** 10
* **Best validation loss:** `1.0549`
* **Validation accuracy at best epoch:** `0.7526`
* **Learning rate:** `1e-3`

The highest validation accuracy was obtained at epoch 7:

* **Validation accuracy:** `0.7628`
* **Validation loss:** `1.1581`

Since checkpoint selection is based on validation loss, epoch 10 is considered the best model for this experiment.

### Learning Rate Behavior

The initial learning rate was `0.01`.

The validation loss became unstable during the first three epochs:

* Epoch 1: `1.1403`
* Epoch 2: `1.5766`
* Epoch 3: `1.8179`

The scheduler reduced the learning rate to `0.001` at epoch 4.

Further reductions occurred during training:

* Epoch 4: `1e-3`
* Epoch 13: `1e-4`
* Epoch 16: `1e-5`
* Epoch 19: `1e-6`

The reduction from `1e-2` to `1e-3` noticeably stabilized training.

---

### Test Set Evaluation

The best model checkpoint was evaluated on the held-out test set containing **733 samples**.

#### Overall Performance

* **Test Accuracy:** `0.7285`
* **Macro F1-score:** `0.56`
* **Weighted F1-score:** `0.73`

#### Classification Report

| Class            | Precision |   Recall | F1-score | Support |
| ---------------- | --------: | -------: | -------: | ------: |
| No DR            |      0.94 |     0.96 | **0.95** |     361 |
| Mild             |      0.52 |     0.62 |     0.57 |      74 |
| Moderate         |      0.65 |     0.52 |     0.58 |     200 |
| Severe           |      0.26 |     0.33 | **0.29** |      39 |
| Proliferative DR |      0.37 |     0.42 |     0.40 |      59 |
| **Macro Avg**    |  **0.55** | **0.57** | **0.56** |     733 |
| **Weighted Avg** |  **0.74** | **0.73** | **0.73** |     733 |

#### Confusion Matrix

```text
[[346  11   4   0   0]
 [  6  46  16   1   5]
 [ 16  28 104  23  29]
 [  0   1  17  13   8]
 [  1   2  18  13  25]]
```

Rows represent actual classes and columns represent predicted classes.

### Test Set Observations

* The model performs very well on the **No DR** class, achieving a recall of `96%` and an F1-score of `0.95`.
* Performance decreases considerably on the minority disease classes.
* The **Severe** class is the weakest class, with an F1-score of only `0.29`.
* Only `13` of the `39` Severe samples were classified correctly.
* The model frequently confuses **Moderate**, **Severe**, and **Proliferative DR** cases.
* Moderate cases are frequently predicted as Mild, Severe, or Proliferative DR.
* The difference between the weighted F1-score (`0.73`) and macro F1-score (`0.56`) highlights the effect of class imbalance.
* Therefore, accuracy and weighted metrics alone do not provide a complete picture of model performance for this dataset.

### Overall Observations

* The model learned rapidly during the first several epochs.
* The high initial learning rate caused instability in validation loss during the first three epochs.
* After the learning rate was reduced to `1e-3`, validation loss became more stable.
* Training loss continued to decrease throughout most of training.
* Validation loss stopped improving consistently after approximately epoch 10.
* This suggests a degree of overfitting.
* The highest validation accuracy (`76.28%`) did not correspond to the lowest validation loss.
* Test-set results confirm that the model performs substantially better on the majority class than on minority disease classes.
* The confusion matrix shows that distinguishing between different stages of diabetic retinopathy remains challenging.

### Conclusion

Using an initial learning rate of `0.01` with `ReduceLROnPlateau` allowed the model to train successfully, but the high initial learning rate caused noticeable instability during the early epochs.

The best validation loss was `1.0549` at epoch 10, with a validation accuracy of `75.26%`.

On the held-out test set, the model achieved an accuracy of `72.85%` and a macro F1-score of `0.56`.

The strong performance on the `No DR` class combined with relatively weak performance on `Severe` and `Proliferative DR` indicates that class imbalance and the similarity between adjacent disease stages are important challenges.

This experiment provides a baseline for the next experiment, where part of the pretrained ResNet50 backbone will be unfrozen and fine-tuned.


## Experiment 02 — ResNet50 Layer4 Fine-Tuning

**Date:** 2026-08-10

### Objective

Evaluate whether fine-tuning the final residual block (`layer4`) of a pretrained ResNet50 improves diabetic retinopathy classification compared with freezing the entire backbone and training only the final fully connected layer.

### Model

* Architecture: ResNet50
* Pretrained weights: ImageNet
* Number of classes: 5
* Backbone: Partially fine-tuned
* Frozen layers: ResNet50 layers before `layer4`
* Trainable layers:

  * `layer4`
  * Final fully connected layer
* Model: `ResNet50_layer4_ft`

### Dataset

* Task: Diabetic Retinopathy Classification
* Number of classes: 5
* Classes:

  * 0 — No DR
  * 1 — Mild
  * 2 — Moderate
  * 3 — Severe
  * 4 — Proliferative DR
* Split strategy: Stratified
* Train/Test split: 80% / 20%
* Train/Validation split: 80% / 20%
* Random state: 10

### Preprocessing

* Images converted to RGB
* Images normalized to `[0, 1]` before calculating statistics
* Training mean and standard deviation calculated only from the training set
* Training augmentation:

  * Random horizontal flip
  * Random rotation
  * Color jitter
* Validation/test augmentation:

  * No augmentation

### Loss

* Loss function: Weighted CrossEntropyLoss
* Class weights: Yes
* Class weights calculated from the training set

### Optimizer

* Optimizer: AdamW
* Initial learning rate: `1e-4`
* Weight decay: `0.001`
* Only parameters with `requires_grad=True` were passed to the optimizer

### Scheduler

* Scheduler: ReduceLROnPlateau
* Mode: `min`
* Factor: `0.1`
* Patience: `2`
* Minimum learning rate: `1e-6`
* Monitored metric: Validation loss

### Training

* Epochs: 20
* Batch size: 32
* Device: CPU

### Training Results

| Epoch | Train Loss |  Train Acc |   Val Loss |    Val Acc |   LR |
| ----: | ---------: | ---------: | ---------: | ---------: | ---: |
|     1 |     1.3495 |     0.5907 |     1.1218 |     0.7133 | 1e-4 |
|     2 |     0.9598 |     0.7200 |     0.9727 |     0.7150 | 1e-4 |
|     3 |     0.7509 |     0.7866 | **0.9116** | **0.7952** | 1e-4 |
|     4 |     0.6201 |     0.8237 |     0.9687 |     0.7526 | 1e-4 |
|     5 |     0.5313 |     0.8378 |     0.9310 |     0.7696 | 1e-4 |
|     6 |     0.4639 |     0.8600 |     1.0355 |     0.7645 | 1e-5 |
|     7 |     0.3297 |     0.8890 |     1.0106 |     0.7765 | 1e-5 |
|     8 |     0.3183 |     0.9082 |     0.9877 |     0.7765 | 1e-5 |
|     9 |     0.3052 |     0.9048 |     1.0212 |     0.7833 | 1e-6 |
|    10 |     0.3107 |     0.9035 |     1.0139 |     0.7782 | 1e-6 |
|    11 |     0.3020 |     0.9095 |     1.0097 |     0.7833 | 1e-6 |
|    12 |     0.2937 |     0.9108 |     1.0176 |     0.7765 | 1e-6 |
|    13 |     0.2894 |     0.9074 |     1.0544 |     0.7765 | 1e-6 |
|    14 |     0.3005 |     0.9142 |     1.0166 |     0.7782 | 1e-6 |
|    15 |     0.2826 |     0.9087 |     1.0230 |     0.7782 | 1e-6 |
|    16 |     0.2952 |     0.9031 |     1.0084 |     0.7747 | 1e-6 |
|    17 |     0.2770 | **0.9245** |     1.0833 |     0.7850 | 1e-6 |
|    18 |     0.2890 |     0.9138 |     1.0362 |     0.7782 | 1e-6 |
|    19 |     0.2707 |     0.9142 |     1.0879 |     0.7713 | 1e-6 |
|    20 |     0.2760 |     0.9163 |     1.0359 |     0.7782 | 1e-6 |

### Best Validation Result

The checkpoint was selected based on the lowest validation loss.

* **Best epoch:** 3
* **Best validation loss:** `0.9116`
* **Validation accuracy at best epoch:** `0.7952`
* **Learning rate:** `1e-4`
* **Best checkpoint:** `resnet50_layer4_ft_best_model.pth`

The highest validation accuracy was also obtained at epoch 3:

* **Validation accuracy:** `79.52%`

After epoch 3, the validation loss did not improve further despite continued reduction of the learning rate.

### Learning Rate Behavior

The initial learning rate was `1e-4`.

The scheduler reduced the learning rate as validation loss stopped improving:

* Epochs 1–5: `1e-4`
* Epochs 6–8: `1e-5`
* Epochs 9–20: `1e-6`

Despite these reductions, validation loss remained around `1.0` after the early improvement at epoch 3.

### Overfitting Observation

The model showed a significant increase in training performance after the third epoch while validation performance stopped improving.

For example:

* Epoch 3:

  * Train Accuracy: `78.66%`
  * Validation Accuracy: `79.52%`
* Epoch 20:

  * Train Accuracy: `91.63%`
  * Validation Accuracy: `77.82%`

This indicates that the model increasingly fitted the training data without obtaining corresponding improvements on the validation set.

Therefore, the best checkpoint occurred very early in training at epoch 3.

---

### Test Set Evaluation

The selected best checkpoint was evaluated on the held-out test set containing **733 samples**.

#### Overall Performance

* **Test Accuracy:** `0.7872` (**78.72%**)
* **Macro Precision:** `0.64`
* **Macro Recall:** `0.61`
* **Macro F1-score:** `0.62`
* **Weighted Precision:** `0.78`
* **Weighted Recall:** `0.79`
* **Weighted F1-score:** `0.78`

#### Classification Report

| Class            | Precision |   Recall | F1-score | Support |
| ---------------- | --------: | -------: | -------: | ------: |
| No DR            |      0.94 | **0.97** | **0.96** |     361 |
| Mild             |      0.64 |     0.49 |     0.55 |      74 |
| Moderate         |      0.69 | **0.74** | **0.72** |     200 |
| Severe           |      0.45 |     0.44 |     0.44 |      39 |
| Proliferative DR |      0.47 |     0.42 |     0.45 |      59 |
| **Macro Avg**    |  **0.64** | **0.61** | **0.62** |     733 |
| **Weighted Avg** |  **0.78** | **0.79** | **0.78** |     733 |

#### Confusion Matrix

```text
[[350   7   4   0   0]
 [  6  36  25   0   7]
 [ 13  10 149  12  16]
 [  0   0  17  17   5]
 [  2   3  20   9  25]]
```

Rows represent actual classes and columns represent predicted classes.

### Test Set Observations

* The model achieved a test accuracy of `78.72%`, substantially higher than the `72.85%` obtained by the frozen-backbone model.
* Macro F1-score increased from `0.56` to `0.62`.
* Weighted F1-score increased from `0.73` to `0.78`.
* The `No DR` class remained the strongest class, with an F1-score of `0.96`.
* The largest improvement was observed for the `Moderate` class:

  * F1-score: `0.58 → 0.72`
  * Recall: `0.52 → 0.74`
* The `Severe` class also improved:

  * F1-score: `0.29 → 0.44`
  * Recall: `0.33 → 0.44`
* Performance on the `Mild` class decreased slightly:

  * F1-score: `0.57 → 0.55`
  * Recall: `0.62 → 0.49`
* The model frequently confused Mild cases with Moderate cases, with `25` Mild samples classified as Moderate.
* The `Severe` and `Proliferative DR` classes remained challenging, with F1-scores of `0.44` and `0.45`, respectively.
* The difference between macro F1 (`0.62`) and weighted F1 (`0.78`) indicates that class imbalance continues to affect overall performance.

### Comparison With Experiment 01

| Metric              | ResNet50 Frozen | ResNet50 Layer4 FT |      Change |
| ------------------- | --------------: | -----------------: | ----------: |
| Test Accuracy       |          0.7285 |         **0.7872** | **+0.0587** |
| Macro Precision     |            0.55 |           **0.64** |       +0.09 |
| Macro Recall        |            0.57 |           **0.61** |       +0.04 |
| Macro F1            |            0.56 |           **0.62** |   **+0.06** |
| Weighted F1         |            0.73 |           **0.78** |       +0.05 |
| No DR F1            |            0.95 |           **0.96** |       +0.01 |
| Mild F1             |            0.57 |               0.55 |       -0.02 |
| Moderate F1         |            0.58 |           **0.72** |   **+0.14** |
| Severe F1           |            0.29 |           **0.44** |   **+0.15** |
| Proliferative DR F1 |            0.40 |           **0.45** |       +0.05 |

### Conclusion

Fine-tuning the final residual block of ResNet50 substantially improved test-set performance compared with freezing the entire backbone.

Test accuracy increased from `72.85%` to `78.72%`, while macro F1-score increased from `0.56` to `0.62`.

The most significant improvements were observed in the Moderate and Severe classes, suggesting that fine-tuning allowed the model to learn more task-specific features for diabetic retinopathy.

However, the model still exhibits considerable difficulty distinguishing Mild, Severe, and Proliferative DR cases. The increasing gap between training and validation performance also indicates overfitting, with the best validation checkpoint occurring at epoch 3.

Overall, **ResNet50 Layer4 Fine-Tuning is currently the better-performing model and provides a strong basis for the next experiment.**


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

The next experiment should focus on strategies that specifically improve minority-class representation and generalization, such as targeted data augmentation, balanced sampling, or other class-balancing techniques before moving to a substantially different model architecture.


## Experiment 04 — ResNet50 Layer4 Fine-Tuning with Focal Loss and WeightedRandomSampler

**Date:** 2026-08-13

### Objective

Evaluate whether combining Focal Loss with a `WeightedRandomSampler` improves the performance of the ResNet50 layer4 fine-tuning model on the imbalanced diabetic retinopathy dataset.

The main goal was to improve performance on minority classes, particularly Severe and Proliferative DR, which remained the weakest classes in previous experiments.

### Model

- Architecture: ResNet50
- Pretrained weights: ImageNet
- Number of classes: 5
- Backbone: Partially fine-tuned
- Fine-tuned layer: Layer4
- Trainable layers: Layer4 and final fully connected layer

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

The dataset is significantly class-imbalanced, with Severe and Proliferative DR being substantially underrepresented compared with No DR and Moderate.

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
- Gamma: `2.0`
- Class weights: No

Focal Loss was used to focus training on harder and misclassified examples without explicitly applying class weights.

### Sampling

- Sampling strategy: `WeightedRandomSampler`
- Applied only to the training set
- Sampling weights calculated from inverse class frequencies
- Sampling with replacement: Yes
- Number of sampled training examples per epoch: Equal to the size of the training set

The purpose of the sampler was to increase the frequency of minority-class samples during training.

### Optimizer

- Optimizer: AdamW
- Initial learning rate: `1e-4`
- Weight decay: `0.001`

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

### Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro F1 | LR |
| ----: | ---------: | --------: | -------: | ------: | ------------: | ---: |
| 1 | 0.7194 | 0.5365 | 0.3956 | 0.7082 | 0.5515 | 1e-4 |
| 2 | 0.3722 | 0.7226 | 0.3085 | 0.7611 | 0.5992 | 1e-4 |
| 3 | 0.2520 | 0.8020 | 0.3384 | 0.7389 | 0.5893 | 1e-4 |
| 4 | 0.1864 | 0.8549 | 0.3048 | 0.7730 | 0.6032 | 1e-4 |
| 5 | 0.1492 | 0.8767 | 0.3157 | 0.7713 | 0.5927 | 1e-4 |
| 6 | 0.1227 | 0.8924 | 0.3282 | 0.7713 | 0.5839 | 1e-4 |
| 7 | 0.0932 | 0.9146 | 0.3329 | 0.7816 | 0.6034 | 1e-5 |
| 8 | 0.0853 | 0.9189 | 0.3148 | 0.7867 | 0.6106 | 1e-5 |
| 9 | 0.0785 | 0.9283 | 0.3347 | 0.7747 | 0.6023 | 1e-5 |
| 10 | 0.0683 | 0.9385 | 0.3268 | 0.7867 | 0.6078 | 1e-6 |
| 11 | 0.0639 | 0.9471 | 0.3340 | 0.7799 | 0.5971 | 1e-6 |
| 12 | 0.0696 | 0.9407 | 0.3257 | 0.7850 | 0.6138 | 1e-6 |
| 13 | 0.0662 | 0.9441 | 0.3290 | 0.7935 | 0.6220 | 1e-6 |
| 14 | 0.0536 | 0.9488 | 0.3278 | 0.7850 | 0.6043 | 1e-6 |
| 15 | 0.0720 | 0.9321 | 0.3280 | 0.7884 | 0.6153 | 1e-6 |
| 16 | 0.0634 | 0.9424 | 0.3288 | 0.7901 | 0.6142 | 1e-6 |
| 17 | 0.0595 | 0.9454 | 0.3280 | 0.7986 | 0.6210 | 1e-6 |
| 18 | 0.0676 | 0.9402 | 0.3319 | 0.7935 | 0.6158 | 1e-6 |
| 19 | 0.0701 | 0.9420 | 0.3372 | 0.7799 | 0.5939 | 1e-6 |
| 20 | 0.0620 | 0.9441 | 0.3265 | 0.7918 | 0.6123 | 1e-6 |

### Best Validation Result

The best checkpoint was selected based on validation Macro F1.

- **Best epoch:** 13
- **Best validation Macro F1:** `0.6220`
- **Validation accuracy at best epoch:** `0.7935`
- **Validation loss:** `0.3290`
- **Learning rate:** `1e-6`

The highest validation accuracy was obtained at epoch 17:

- **Validation accuracy:** `0.7986`
- **Validation Macro F1:** `0.6210`

Since checkpoint selection is based on Macro F1, epoch 13 was selected as the best model for this experiment.

### Training Behavior

The model learned very quickly and achieved high training performance:

- Training accuracy increased from `53.65%` at epoch 1 to `94.41%` at epoch 20.
- Training loss decreased from `0.7194` to `0.0620`.

However, validation performance plateaued relatively early.

The best validation Macro F1 was obtained at epoch 13 (`0.6220`), while training accuracy had already reached approximately `94%`.

This indicates substantial overfitting and limited generalization improvement from continued training.

### Comparison with Previous Focal Loss Experiment

The previous experiment using ResNet50 Layer4 fine-tuning with Focal Loss but without weighted sampling achieved the following test results:

| Metric | Focal Loss | Focal Loss + Weighted Sampler |
|---|---:|---:|
| Test Accuracy | `0.8213` | Not evaluated yet |
| Macro F1 | `0.65` | Validation: `0.6220` |
| Mild F1 | `0.63` | Not evaluated yet |
| Moderate F1 | `0.79` | Not evaluated yet |
| Severe F1 | `0.33` | Not evaluated yet |
| Proliferative DR F1 | `0.53` | Not evaluated yet |

The validation results do not indicate a clear improvement over the previous Focal Loss configuration.

### Observations

- Weighted sampling increased the exposure of minority classes during training.
- Training accuracy increased rapidly, reaching approximately `94%`.
- Validation Macro F1 improved only modestly and remained around `0.60–0.62`.
- The best validation Macro F1 (`0.6220`) was lower than the previous model's test Macro F1 (`0.65`).
- Validation accuracy reached `79.86%`, but this did not correspond to the best Macro F1.
- The model continued fitting the training data while validation performance remained relatively stable.
- This suggests that class imbalance is not the only problem; overfitting and limited dataset size also contribute significantly.
- WeightedRandomSampler alone did not provide sufficient evidence of improved generalization.
- Severe and Proliferative DR remain the main challenging classes.

### Conclusion

Adding `WeightedRandomSampler` to Focal Loss did not produce a clear improvement over the previous Focal Loss configuration.

Although the sampler was intended to improve minority-class learning, the validation Macro F1 remained limited at `0.6220`, while training accuracy increased to more than `94%`.

Therefore, the current experiment does not justify keeping `WeightedRandomSampler` as the default training strategy.

The previous `ResNet50_layer4_ft_focal` configuration remains the stronger baseline.

The next experiment should focus on improving generalization rather than simply increasing the sampling frequency of minority classes. Stronger and more appropriate data augmentation is a reasonable next step, particularly because the dataset is relatively small and heavily imbalanced.