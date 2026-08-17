# Experiment 9 — DenseNet121 Fine-Tuning

## Objective

Evaluate DenseNet121 as an alternative CNN architecture for diabetic retinopathy classification and compare it with the current ResNet50 baseline.

## Model

**Model:** `DenseNet121_denseblock4_ft_focal`

- Pretrained `DenseNet121_Weights.DEFAULT`
- All parameters frozen initially
- Fine-tuned `features.denseblock4`
- Fine-tuned `features.norm5`
- Replaced the classifier with a 5-class linear layer
- Focal Loss with `gamma = 2.0`

## Training Configuration

| Parameter | Value |
|---|---|
| Architecture | DenseNet121 |
| Fine-tuned layers | `denseblock4`, `norm5` |
| Number of classes | 5 |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | AdamW |
| Learning rate | `5e-5` |
| Weight decay | `1e-3` |
| Loss | Focal Loss |
| Focal gamma | 2.0 |
| Weighted sampler | No |
| Scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.1 |
| Scheduler patience | 2 |
| Minimum LR | `1e-6` |
| Device | CPU |

The existing dataset split and augmentation pipeline were retained.

## Training Results

The best validation Macro F1 was obtained at **Epoch 8**:

- Train Loss: `0.2098`
- Train Accuracy: `0.8361`
- Validation Loss: `0.2810`
- Validation Accuracy: `0.8123`
- Validation Macro F1: **0.6931**
- Learning Rate: `5e-5`

Final epoch:

- Train Loss: `0.0966`
- Train Accuracy: `0.9099`
- Validation Loss: `0.2491`
- Validation Accuracy: `0.8225`
- Validation Macro F1: `0.6671`
- Learning Rate: `1e-6`

The best checkpoint was selected using the highest validation Macro F1.

## Test Results

**Test Accuracy:** `0.7503`

**Test Macro F1:** `0.58`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| No DR | 0.96 | 0.97 | 0.97 | 361 |
| Mild | 0.68 | 0.49 | 0.57 | 74 |
| Moderate | 0.71 | 0.57 | 0.63 | 200 |
| Severe | 0.33 | 0.31 | 0.32 | 39 |
| Proliferative DR | 0.31 | 0.61 | 0.41 | 59 |
| **Macro Average** | **0.60** | **0.59** | **0.58** | **733** |

### Confusion Matrix

[[351   9   1   0   0]
 [  7  36  24   0   7]
 [  7   7 115  16  55]
 [  0   0   9  12  18]
 [  0   1  14   8  36]]


## Analysis

DenseNet121 performed substantially worse than the current ResNet50 baseline.

The model performed very well on the majority class:

- No DR F1: `0.97`

However, minority-class performance remained weak:

- Mild F1: `0.57`
- Moderate F1: `0.63`
- Severe F1: `0.32`
- Proliferative DR F1: `0.41`

A major source of error was confusion between Moderate and Proliferative DR:

- 55 Moderate samples were classified as Proliferative DR.
- 14 Proliferative DR samples were classified as Moderate.

There was also a noticeable gap between validation and test performance. Although validation Macro F1 reached `0.6931`, test Macro F1 was only `0.58`.

## Comparison With Previous Experiments

| Experiment | Model | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|
| Experiment 7 | ResNet50 + layer3/layer4 + BN frozen + Focal Loss | **0.8417** | **0.70** |
| Experiment 8 | ResNet50 + layer3/layer4 + BN frozen + Focal Loss + Weighted Sampler | 0.8404 | **0.70** |
| Experiment 9 | DenseNet121 + denseblock4/norm5 fine-tuning + Focal Loss | 0.7503 | **0.58** |

## Conclusion

Under the current dataset, augmentation pipeline, loss function, and training configuration, **DenseNet121 did not outperform the ResNet50 baseline**.

Experiment 7 remains the strongest configuration, with test accuracy `0.8417` and test Macro F1 approximately `0.70`.

Further experiments will therefore continue from the **ResNet50 architecture**, while DenseNet121 will not be pursued further under this setup.