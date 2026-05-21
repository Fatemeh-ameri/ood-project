import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data import Subset

import matplotlib.pyplot as plt
import numpy as np

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# CIFAR-10 classes
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Unknown classes
unknown_class_names = [
    "airplane",
    "automobile",
    "ship",
    "truck"
]

unknown_class_ids = [
    classes.index(name)
    for name in unknown_class_names
]

# Transform
transform = transforms.ToTensor()

# Test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=False,
    download=False,
    transform=transform
)

# Unknown subset only
unknown_indices = [
    i for i, (_, label)
    in enumerate(test_dataset)
    if label in unknown_class_ids
]

unknown_dataset = Subset(
    test_dataset,
    unknown_indices
)

# DataLoader
unknown_loader = DataLoader(
    unknown_dataset,
    batch_size=64,
    shuffle=False
)

# CNN model
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(32 * 8 * 8, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)

# Load model
model.load_state_dict(
    torch.load(
        "models/known_only_cnn.pth",
        map_location=device
    )
)

model = model.to(device)

print("Known-only model loaded!")

# Evaluation mode
model.eval()

true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in unknown_loader:
        images = images.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.cpu().tolist())

# Confusion matrix:
# rows = true unknown classes
# columns = predicted CIFAR-10 classes
cm = np.zeros(
    (len(unknown_class_names), len(classes)),
    dtype=int
)

for true_label, predicted_label in zip(true_labels, predicted_labels):
    row = unknown_class_ids.index(true_label)
    col = predicted_label
    cm[row, col] += 1

print("Rows:", unknown_class_names)
print("Columns:", classes)
print("Confusion Matrix:")

for row in cm:
    print(row.tolist())

# Plot
plt.figure(figsize=(10, 5))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    range(len(classes)),
    classes,
    rotation=45
)

plt.yticks(
    range(len(unknown_class_names)),
    unknown_class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("Predictions for Unknown Classes")

plt.tight_layout()
plt.show()