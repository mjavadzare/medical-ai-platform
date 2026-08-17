# Experiment 08 — Weighted Random Sampler

## Objective

Evaluate whether using a WeightedRandomSampler during training can improve performance on the imbalanced diabetic retinopathy dataset, particularly for minority classes.

The experiment uses the best configuration from Experiment 07 and adds weighted sampling to the training DataLoader.


## Configuration

### Model

- Architecture: ResNet50
- Pretrained weights: ImageNet
- Fine-tuned layers:
  - layer3
  - layer4
- BatchNorm layers: Frozen
- Classification head: Replaced with a 5-class linear layer

### Loss

- Focal Loss
- Gamma: 2.0

### Class imbalance

Weighted Random Sampling was enabled:

    WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

Class weights were calculated as the inverse of the training-set class frequency:

    class_weights = 1.0 / class_counts.float()

Each training sample was then assigned the weight corresponding to its class.

### Training

- Batch size: 32
- Epochs: 20
- Optimizer: AdamW
- Learning rate: 5e-5
- Weight decay: 1e-3
- Scheduler: ReduceLROnPlateau
- Scheduler factor: 0.1
- Scheduler patience: 2
- Minimum learning rate: 1e-6
- Augmentation: unchanged from previous experiments
- num_workers: 0

### Model selection

The best model was selected according to Validation Macro F1, rather than validation loss or accuracy.


## Training Results

Best validation Macro F1:

- Epoch: 15
- Validation Macro F1: 0.6872
- Validation Accuracy: 0.8328
- Validation Loss: 0.3166
- Learning Rate: 1e-6

Training continued until epoch 20.

The training accuracy increased to approximately 0.94–0.95, while validation Macro F1 fluctuated around 0.65–0.69, indicating that the model continued to show signs of overfitting.


## Test Results

### Overall Performance

- Test Accuracy: 0.8404
- Test Macro F1: 0.696
- Weighted F1: 0.84

### Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.98 | 0.98 | 0.98 | 361 |
| Mild | 0.63 | 0.62 | 0.63 | 74 |
| Moderate | 0.77 | 0.83 | 0.80 | 200 |
| Severe | 0.53 | 0.44 | 0.48 | 39 |
| Proliferative DR | 0.63 | 0.56 | 0.59 | 59 |
| Macro Avg | 0.71 | 0.69 | 0.70 | 733 |

### Confusion Matrix

    [[353   7   1   0   0]
     [  4  46  23   0   1]
     [  1  16 167   9   7]
     [  0   0  11  17  11]
     [  1   4  15   6  33]]


## Comparison with Experiment 07

| Metric | Exp. 07 | Exp. 08 |
|---|---:|---:|
| Test Accuracy | 0.8417 | 0.8404 |
| Test Macro F1 | 0.698 | 0.696 |
| No DR F1 | 0.98 | 0.98 |
| Mild F1 | 0.66 | 0.63 |
| Moderate F1 | 0.79 | 0.80 |
| Severe F1 | 0.47 | 0.48 |
| Proliferative DR F1 | 0.59 | 0.59 |


## Analysis

Adding WeightedRandomSampler did not produce an overall improvement over Experiment 07.

Test Macro F1 decreased slightly:

    0.698 → 0.696

and test accuracy also decreased slightly:

    0.8417 → 0.8404

The sampler did provide small changes for some minority classes. Moderate and Severe F1 improved slightly, while Mild F1 decreased. Proliferative DR remained approximately unchanged.

Therefore, the weighted sampler did not provide a sufficiently meaningful improvement to justify replacing the Experiment 07 configuration.


## Conclusion

Experiment 08 is not selected as the best configuration.

The current best configuration remains:

ResNet50 + layer3/layer4 fine-tuning + frozen BatchNorm + Focal Loss

from Experiment 07, with:

- Test Accuracy: 0.8417
- Test Macro F1: 0.698

Weighted Random Sampling will therefore not be retained for the current configuration.

The experiment nevertheless confirms that simply compensating for class imbalance at the sampling level does not substantially improve the overall performance of the current model.