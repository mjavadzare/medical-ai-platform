## Experiment 07 — ResNet50 Layer3 + Layer4 Fine-Tuning with Frozen BatchNorm

### Objective

Evaluate whether freezing Batch Normalization layers while fine-tuning ResNet50 `layer3` and `layer4` improves generalization and classification performance on the diabetic retinopathy dataset.

### Model

- Architecture: ResNet50
- Pretrained weights: ImageNet
- Fine-tuned layers:
  - `layer3`
  - `layer4`
- Frozen layers:
  - `conv1`
  - `bn1`
  - `layer1`
  - `layer2`
- Batch Normalization:
  - BatchNorm layers inside `layer3` and `layer4` were kept frozen
  - BatchNorm parameters were excluded from gradient updates
  - BatchNorm layers were kept in evaluation mode
- Classifier:
  - Replaced the original ResNet50 classifier with a 5-class linear layer

### Training Configuration

- Epochs: 20
- Batch size: 32
- Optimizer: AdamW
- Learning rate: `5e-5`
- Weight decay: `1e-3`
- Loss function: Focal Loss
- Focal gamma: `2.0`
- Learning-rate scheduler: ReduceLROnPlateau
- Scheduler factor: `0.1`
- Scheduler patience: `2`
- Minimum learning rate: `1e-6`
- Weighted sampler: Disabled
- Data augmentation: Same as previous experiments
- Input size: `224 × 224`
- Device: CPU

### Model Selection

The best model was selected based on **validation Macro F1**, rather than validation loss.

This criterion was kept because the dataset is highly imbalanced and overall validation accuracy/loss can be dominated by the majority class (`No DR`). Macro F1 gives equal importance to all five classes.

### Validation Results

Best validation Macro F1:

- Macro F1: **0.6900**
- Best model was reached before the final training epochs.

The model was then evaluated once on the held-out test set.

### Test Results

- Test Accuracy: **0.8417**
- Macro F1: **0.68**
- Weighted F1: **0.84**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.98 | 0.98 | 0.98 | 361 |
| Mild | 0.67 | 0.65 | 0.66 | 74 |
| Moderate | 0.74 | 0.85 | 0.79 | 200 |
| Severe | 0.52 | 0.44 | 0.47 | 39 |
| Proliferative DR | 0.72 | 0.49 | 0.59 | 59 |
| **Macro Avg** | **0.73** | **0.68** | **0.70** | **733** |
| **Weighted Avg** | **0.84** | **0.84** | **0.84** | **733** |

### Confusion Matrix

[[353   7   1   0   0]
 [  4  48  21   0   1]
 [  1  15 170   9   5]
 [  0   0  17  17   5]
 [  1   2  20   7  29]]

 ### Comparison with Previous Experiment

Compared with Experiment 06 (ResNet50 with `layer3` + `layer4` fine-tuning without explicitly freezing BatchNorm):

| Metric | Experiment 06 | Experiment 07 | Change |
|---|---:|---:|---:|
| Test Accuracy | 0.8090 | **0.8417** | **+3.27 pp** |
| Macro F1 | 0.64 | **0.70** | **+0.06** |

### Analysis

Freezing BatchNorm layers produced a substantial improvement over the previous `layer3 + layer4` fine-tuning configuration.

The improvement is especially important because the dataset is relatively small and imbalanced. Updating BatchNorm statistics during fine-tuning can make the pretrained representation less stable, particularly when batch statistics are noisy or minority classes are underrepresented.

The model achieved strong performance on the majority class (`No DR`) while also improving the minority-class performance compared with previous experiments.

The main remaining weaknesses are:

- `Severe DR` remains difficult to classify.
- `Proliferative DR` still has relatively low recall.
- A considerable number of `Mild` samples are classified as `Moderate`.
- A considerable number of `Severe` and `Proliferative DR` samples are classified as `Moderate`.

### Conclusion

Experiment 07 is currently the **best-performing model configuration**.

The combination of:

- ResNet50 pretrained weights
- Fine-tuning `layer3` and `layer4`
- Frozen BatchNorm layers
- Focal Loss
- Macro F1-based model selection

resulted in a significant improvement over the previous experiments.

This configuration will be used as the current baseline for the next experiment.
