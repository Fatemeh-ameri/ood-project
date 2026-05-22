import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data import Subset

import matplotlib.pyplot as plt

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

# Known classes
known_class_names = [
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse"
]

# Unknown classes
unknown_class_names = [
    "airplane",
    "automobile",
    "ship",
    "truck"
]

known_class_ids = [
    classes.index(name)
    for name in known_class_names
]

unknown_class_ids = [
    classes.index(name)
    for name in unknown_class_names
]

print("Known classes:", known_class_names)
print("Unknown classes:", unknown_class_names)

# Same transform used during ResNet18 training
transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# Test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=False,
    download=False,
    transform=transform
)

# Known subset
known_indices = [
    i for i, (_, label)
    in enumerate(test_dataset)
    if label in known_class_ids
]

# Unknown subset
unknown_indices = [
    i for i, (_, label)
    in enumerate(test_dataset)
    if label in unknown_class_ids
]

known_dataset = Subset(
    test_dataset,
    known_indices
)

unknown_dataset = Subset(
    test_dataset,
    unknown_indices
)

# DataLoaders
known_loader = DataLoader(
    known_dataset,
    batch_size=64,
    shuffle=False
)

unknown_loader = DataLoader(
    unknown_dataset,
    batch_size=64,
    shuffle=False
)

# ResNet18 model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

# Load known-only ResNet18
model.load_state_dict(
    torch.load(
        "models/known_only_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)

print("Known-only ResNet18 loaded!")

model.eval()

known_confidences = []
unknown_confidences = []

# Known confidence
with torch.no_grad():
    for images, _ in known_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        known_confidences.extend(
            confidence.cpu().tolist()
        )

# Unknown confidence
with torch.no_grad():
    for images, _ in unknown_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        unknown_confidences.extend(
            confidence.cpu().tolist()
        )

avg_known = sum(known_confidences) / len(known_confidences)
avg_unknown = sum(unknown_confidences) / len(unknown_confidences)

print(f"Average known confidence:   {avg_known:.3f}")
print(f"Average unknown confidence: {avg_unknown:.3f}")

# Histogram
plt.figure(figsize=(8, 5))

plt.hist(
    known_confidences,
    bins=30,
    alpha=0.7,
    label="Known Classes"
)

plt.hist(
    unknown_confidences,
    bins=30,
    alpha=0.7,
    label="Unknown Classes"
)

plt.xlabel("Confidence")
plt.ylabel("Count")

plt.title("ResNet18 Known vs Unknown Confidence")

plt.legend()

plt.tight_layout()


plt.savefig(
    "reports/figures/resnet18_known_vs_unknown_confidence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()