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
