# Experiment Log

This file tracks the main experiments in the project and the changes tested during model development.

## 1. Baseline Models on CIFAR-10

| Experiment | Model | Setup | Test Accuracy |
|---|---|---|---:|
| 1 | Fully Connected Neural Network | CIFAR-10, all classes, 3 epochs | 43.36% |
| 2 | Simple CNN | CIFAR-10, all classes, 5 epochs | 64.05% |
| 3 | ResNet18 | CIFAR-10, all classes, normalization, 10 epochs | 76.47% |

## 2. Improving ResNet18 on CIFAR-10

| Experiment | Change Tested | Test Accuracy | Notes |
|---|---|---:|---|
| 4 | Added data augmentation | 78.28% | Random crop and horizontal flip were added to the training transform. |
| 5 | Increased training from 10 to 20 epochs | 81.77% | Accuracy improved with longer training. |
| 6 | Added StepLR scheduler | 83.07% | Learning rate was reduced after epoch 10. |
| 7 | Replaced Adam with SGD + momentum | 80.97% | This setup performed worse than Adam in the current configuration. |

Current best full CIFAR-10 model:

| Model | Setup | Test Accuracy |
|---|---|---:|
| ResNet18 | Augmentation, Adam, StepLR, 20 epochs | 83.07% |

## 3. Known-Only Training Setup

Known classes:

- bird
- cat
- deer
- dog
- frog
- horse

Unknown classes:

- airplane
- automobile
- ship
- truck

## 4. Known-Only Model Accuracy

| Model | Setup | Known-Class Accuracy |
|---|---|---:|
| Simple CNN | Known classes only | 61.02% |
| ResNet18 | Known classes only, 10 epochs | 71.33% |
| ResNet18 | Known classes only, augmentation, Adam, StepLR, 20 epochs | 79.70% |

## 5. Known vs Unknown Confidence

| Model | Setup | Known Confidence | Unknown Confidence |
|---|---|---:|---:|
| Simple CNN | Known classes only | 0.631 | 0.552 |
| ResNet18 | Known classes only, 10 epochs | 0.861 | 0.753 |
| ResNet18 | Known classes only, improved recipe | 0.829 | 0.661 |

The improved ResNet18 model increased known-class accuracy and reduced average confidence on unknown samples.

## 6. MSP Thresholding

Maximum Softmax Probability (MSP) was used as a simple baseline for unknown detection.

A sample is treated as unknown when its maximum softmax confidence is below the selected threshold.

### Simple CNN

| Threshold | Known Accepted | Unknown Rejected |
|---:|---:|---:|
| 0.4 | 84.98% | 22.68% |
| 0.5 | 68.57% | 45.58% |
| 0.6 | 51.60% | 63.90% |
| 0.7 | 37.98% | 78.08% |
| 0.8 | 25.47% | 88.05% |

### ResNet18 before improved training recipe

| Threshold | Known Accepted | Unknown Rejected |
|---:|---:|---:|
| 0.4 | 98.38% | 3.83% |
| 0.5 | 94.33% | 14.38% |
| 0.6 | 86.78% | 26.75% |
| 0.7 | 79.50% | 38.98% |
| 0.8 | 71.37% | 51.88% |
| 0.9 | 61.08% | 66.20% |

### ResNet18 after improved training recipe

| Threshold | Known Accepted | Unknown Rejected |
|---:|---:|---:|
| 0.4 | 97.08% | 9.82% |
| 0.5 | 91.13% | 26.95% |
| 0.6 | 82.42% | 43.45% |
| 0.7 | 74.03% | 57.23% |
| 0.8 | 65.35% | 69.45% |
| 0.9 | 53.37% | 83.03% |

## 7. Notes

- ResNet18 improved known-class accuracy compared with the Simple CNN.
- Better classification accuracy did not automatically remove overconfidence on unknown samples.
- The improved ResNet18 recipe reduced average unknown confidence compared with the earlier ResNet18 setup.
- MSP thresholding shows a clear trade-off between accepting known samples and rejecting unknown samples.
- The next step is to save key plots and add selected figures to the README or reports.