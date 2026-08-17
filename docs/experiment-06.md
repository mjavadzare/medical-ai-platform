## Experiment 06 — EfficientNet-B0 + Focal Loss

### Objective

Evaluate EfficientNet-B0 as an alternative backbone to ResNet50
while keeping the existing training pipeline and Focal Loss configuration.

### Configuration

- Model: EfficientNet-B0
- Pretrained weights: ImageNet
- Backbone: Frozen
- Trainable layers: Classifier only
- Number of classes: 5
- Batch size: 32
- Epochs: 20
- Optimizer: AdamW
- Learning rate: 5e-5
- Weight decay: 1e-3
- Loss: Focal Loss
- Focal gamma: 2.0
- Scheduler: ReduceLROnPlateau
- Scheduler factor: 0.1
- Scheduler patience: 2
- Minimum learning rate: 1e-6
- Device: CPU
- Best model selection: Validation Macro F1

### Training Results

Best validation result:

- Validation Accuracy: 0.7457
- Validation Macro F1: 0.4244
- Best epoch: 18

Training accuracy at epoch 20:

- Train Accuracy: 0.7251
- Train Loss: 0.4242

### Test Results

- Test Accuracy: 0.7190
- Macro F1: 0.40
- Weighted F1: 0.67

Per-class F1:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| No DR | 0.85 | 0.97 | 0.91 |
| Mild | 0.55 | 0.24 | 0.34 |
| Moderate | 0.55 | 0.78 | 0.64 |
| Severe | 0.00 | 0.00 | 0.00 |
| Proliferative DR | 1.00 | 0.07 | 0.13 |

### Confusion Matrix

[[349   2  10   0   0]
 [ 20  18  36   0   0]
 [ 32  12 156   0   0]
 [  1   0  38   0   0]
 [  7   1  45   2   4]]

### Comparison with Best ResNet50 Model

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| ResNet50 layer3+layer4 + Focal Loss | 0.8090 | 0.64 | 0.80 |
| EfficientNet-B0 + Focal Loss | 0.7190 | 0.40 | 0.67 |

### Analysis

EfficientNet-B0 performed substantially worse than the current ResNet50
baseline.

The model achieved a test accuracy of 71.90% and a Macro F1 of 0.40.
The main weakness was the recognition of minority and more severe disease
classes.

The Severe class was completely missed:

- Precision: 0.00
- Recall: 0.00
- F1: 0.00

The model also showed a strong tendency to classify Severe and Proliferative
DR samples as Moderate.

In particular, 38 out of 39 Severe test samples were classified as Moderate.

The relatively low training accuracy (72.51% at epoch 20) suggests that the
frozen EfficientNet-B0 backbone did not provide sufficiently useful features
for this task under the current configuration.

### Conclusion

EfficientNet-B0 with a frozen backbone and classifier-only fine-tuning
performed significantly worse than the current ResNet50 model.

Therefore, EfficientNet-B0 is not selected as the current best model.

The current best configuration remains:

**ResNet50 layer3 + layer4 fine-tuning + Focal Loss**

Further EfficientNet-B0 fine-tuning may be investigated in the future, but
given the current CPU-only training environment, the next experiment will
focus on evaluating a different backbone architecture.