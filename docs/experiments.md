# Experiments

This file records the experiments performed during model development.

The purpose is to keep track of training configurations, model changes, evaluation results, and observations.

---

## Experiment 01 — ResNet50 Baseline

**Date:** 2026-08-07

### Model

* Architecture: ResNet50
* Pretrained weights: ImageNet
* Number of classes: 5
* Frozen backbone: Yes
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
* Split: Stratified
* Test size: 20%
* Validation size: 20%
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

* Loss function: CrossEntropyLoss
* Class weights: Yes
* Class weights calculated from the training set

### Optimizer

* Optimizer: AdamW
* Learning rate: `0.001`
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

### Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |   LR |
| ----: | ---------: | --------: | -------: | ------: | ---: |
|     1 |     1.3938 |    0.5408 |   1.2872 |  0.7304 | 1e-3 |
|     2 |     1.1568 |    0.6829 |   1.1514 |  0.6775 | 1e-3 |
|     3 |     1.0801 |    0.6965 |   1.1120 |  0.7509 | 1e-3 |
|     4 |     1.0076 |    0.7273 |   1.0721 |  0.7372 | 1e-3 |
|     5 |     0.9823 |    0.7200 |   1.0797 |  0.7679 | 1e-3 |
|     6 |     0.9285 |    0.7499 |   1.0294 |  0.7338 | 1e-3 |
|     7 |     0.9069 |    0.7469 |   1.0359 |  0.7526 | 1e-3 |
|     8 |     0.9054 |    0.7495 |   1.0008 |  0.7372 | 1e-3 |
|     9 |     0.8683 |    0.7486 |   1.0133 |  0.7560 | 1e-3 |
|    10 |     0.8394 |    0.7678 |   1.0129 |  0.7457 | 1e-3 |
|    11 |     0.8324 |    0.7665 |   1.0082 |  0.6655 | 1e-4 |
|    12 |     0.8038 |    0.7734 |   0.9949 |  0.6980 | 1e-4 |
|    13 |     0.7815 |    0.7849 |   0.9895 |  0.7321 | 1e-4 |
|    14 |     0.8117 |    0.7857 |   0.9855 |  0.7321 | 1e-4 |
|    15 |     0.8027 |    0.7755 |   0.9855 |  0.7372 | 1e-4 |
|    16 |     0.7980 |    0.7845 |   0.9894 |  0.7321 | 1e-4 |
|    17 |     0.8006 |    0.7853 |   0.9925 |  0.7270 | 1e-5 |
|    18 |     0.7857 |    0.7870 |   0.9836 |  0.7321 | 1e-5 |
|    19 |     0.7969 |    0.7785 |   0.9900 |  0.7474 | 1e-5 |
|    20 |     0.7995 |    0.7853 |   0.9931 |  0.7389 | 1e-5 |

### Best Validation Result

* Best validation loss: `0.9836`
* Best epoch: `18`
* Validation accuracy at best loss: `0.7321`
* Best checkpoint: `resnet50_best_checkpoint.pth`

### Observations

* The model learns quickly during the first several epochs.
* Validation loss improves more slowly than training loss.
* The learning rate was reduced from `1e-3` to `1e-4` after validation loss stopped improving.
* A second reduction to `1e-5` occurred later in training.
* Training accuracy reached approximately `78.7%`, while validation accuracy remained around `73%`.
* The gap between training and validation performance suggests some degree of overfitting.
* Because the dataset is class-imbalanced, accuracy alone is not sufficient for evaluating the model.


## Experiment 02 — ResNet50 with High Learning Rate

**Date:** 2026-08-07

### Objective

Evaluate the effect of using a higher initial learning rate compared with the baseline experiment.

### Model

* Architecture: ResNet50
* Pretrained weights: ImageNet
* Number of classes: 5
* Backbone: Frozen
* Trainable layer: Final fully connected layer

### Dataset

* Task: Diabetic Retinopathy Classification
* Number of classes: 5
* Stratified train/validation/test split
* Test size: 20%
* Validation size: 20%
* Random state: 10

### Loss

* Loss function: Weighted CrossEntropyLoss
* Class weights: Calculated from the training set

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

### Results

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

The best validation loss was obtained at:

* **Epoch:** 10
* **Validation loss:** `1.0549`
* **Validation accuracy:** `0.7526`
* **Learning rate:** `1e-3`

However, the highest validation accuracy was:

* **Epoch:** 7
* **Validation accuracy:** `0.7628`
* **Validation loss:** `1.1581`

Therefore, if the checkpoint selection criterion is validation loss, **epoch 10** is the correct best checkpoint.

### Observations

The initial learning rate of `0.01` was relatively aggressive.

During the first three epochs, the validation loss became unstable:

* Epoch 1: `1.1403`
* Epoch 2: `1.5766`
* Epoch 3: `1.8179`

At epoch 4, the scheduler reduced the learning rate from `1e-2` to `1e-3`. After this reduction, training became considerably more stable and validation loss started to decrease.

The model continued improving after the learning-rate reduction and reached its lowest validation loss of `1.0549` at epoch 10.

After approximately epoch 10, training loss continued to decrease while validation loss stopped improving consistently. This indicates that the model was increasingly fitting the training data without obtaining a corresponding improvement on the validation set.

The learning rate was subsequently reduced to:

* `1e-4` at epoch 13
* `1e-5` at epoch 16
* `1e-6` at epoch 19

These additional reductions did not produce a meaningful improvement in validation loss.

### Comparison With Experiment 01

| Metric          | Experiment 01 | Experiment 02 |
| --------------- | ------------: | ------------: |
| Initial LR      |        `1e-3` |        `1e-2` |
| Best Val Loss   |      `0.9836` |      `1.0549` |
| Best Val Acc    |     `0.7833`* |      `0.7628` |
| Final Train Acc |      `0.7853` |      `0.8092` |
| Final Val Acc   |      `0.7389` |      `0.7491` |

* Experiment 01's best checkpoint was selected based on validation loss, not validation accuracy.

### Conclusion

Increasing the initial learning rate from `1e-3` to `1e-2` did **not** improve the overall validation loss compared with Experiment 01.

The higher learning rate caused noticeable instability during the first few epochs. Although the scheduler eventually reduced the learning rate and stabilized training, the best validation loss remained worse than the baseline.

Therefore, based on validation loss, **Experiment 01 with an initial learning rate of `1e-3` performed better than Experiment 02**.

### Next Experiment

The next experiment should investigate fine-tuning part of the pretrained ResNet50 backbone instead of keeping the entire backbone frozen.

A smaller learning rate should be used for the pretrained layers, while the classifier can use a higher learning rate.
