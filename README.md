# OOD Research Project

This project is a small research-oriented exploration of image classification confidence and out-of-distribution (OOD) behavior using PyTorch.

The current experiments use CIFAR-10 and a simple convolutional neural network (CNN).

## Current Work

* Loading and visualizing CIFAR-10
* Training simple neural network models
* Training and evaluating a CNN baseline
* Saving and loading trained models
* Measuring prediction confidence with softmax
* Comparing confidence for correct and incorrect predictions
* Visualizing confidence distributions

## Current Results

| Model                          | Test Accuracy |
| ------------------------------ | ------------- |
| Fully Connected Neural Network | 43.36%        |
| Simple CNN                     | 64.05%        |

## Confidence Analysis

Average confidence scores:

| Prediction Type     | Average Confidence |
| ------------------- | ------------------ |
| Correct predictions | 0.739              |
| Wrong predictions   | 0.530              |

The model is usually more confident when predictions are correct, but there is still overlap between correct and incorrect predictions.

## Next Steps

* Train on selected CIFAR-10 classes only
* Treat remaining classes as unseen data
* Compare confidence on known vs unknown classes
* Explore simple OOD detection behavior

## Tools

* Python
* PyTorch
* Torchvision
* Matplotlib
* Git / GitHub
