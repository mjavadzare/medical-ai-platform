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
