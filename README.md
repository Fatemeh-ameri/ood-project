# OOD Research Project

This project is a small research-oriented exploration of image classification confidence and out-of-distribution (OOD) behavior using PyTorch.

The experiments use CIFAR-10, starting from simple neural network baselines and moving toward known vs unknown class evaluation.

## Current Work

- Loading and visualizing CIFAR-10
- Training and evaluating baseline image classification models
- Saving and loading trained model checkpoints
- Measuring prediction confidence with softmax
- Comparing confidence for correct and incorrect predictions
- Training models on selected known classes only
- Treating the remaining classes as unknown during evaluation
- Testing Maximum Softmax Probability (MSP) as a simple OOD baseline

## Current Results

| Model | Training Setup | Test Accuracy |
|---|---|---:|
| Fully Connected Neural Network | CIFAR-10, all classes | 43.36% |
| Simple CNN | CIFAR-10, all classes | 64.05% |
| ResNet18 | CIFAR-10, all classes | 76.47% |
| Simple CNN | Known classes only | 61.02% |
| ResNet18 | Known classes only | 71.33% |

## Confidence Analysis

Average confidence scores for the Simple CNN trained on all CIFAR-10 classes:

| Prediction Type | Average Confidence |
|---|---:|
| Correct predictions | 0.739 |
| Wrong predictions | 0.530 |

The model is usually more confident when predictions are correct, but there is still overlap between correct and incorrect predictions.

## Known vs Unknown Experiment

For the OOD-style experiment, the models were trained only on six CIFAR-10 animal classes:

- bird
- cat
- deer
- dog
- frog
- horse

The remaining vehicle classes were treated as unknown during evaluation:

- airplane
- automobile
- ship
- truck

Average softmax confidence:

| Model | Known Confidence | Unknown Confidence |
|---|---:|---:|
| Simple CNN | 0.631 | 0.552 |
| ResNet18 | 0.861 | 0.753 |

ResNet18 achieved better known-class accuracy, but it also remained highly confident on many unknown samples.

## MSP Thresholding

Maximum Softmax Probability (MSP) was used as a simple baseline for unknown detection.

A sample is treated as unknown when its maximum softmax confidence is below a selected threshold.

At threshold `0.8`:

| Model | Known Accepted | Unknown Rejected |
|---|---:|---:|
| Simple CNN | 25.47% | 88.05% |
| ResNet18 | 71.37% | 51.88% |

This shows a trade-off between accepting known samples and rejecting unknown samples. ResNet18 keeps more known samples accepted, but it rejects fewer unknown samples at the same threshold.

## Observations

- ResNet18 performs better than the Simple CNN on known-class classification.
- Higher classification accuracy does not automatically lead to better OOD rejection with softmax confidence.
- Simple CNN has lower confidence overall, so it rejects more unknown samples at the same MSP threshold.
- ResNet18 is more confident on both known and unknown samples.

## Next Steps

- Save plots automatically for README and reports
- Add a cleaner comparison table for MSP threshold sweeps
- Test energy-based OOD scoring
- Explore calibration methods such as temperature scaling

## Tools

- Python
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- Git / GitHub