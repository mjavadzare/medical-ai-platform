
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
