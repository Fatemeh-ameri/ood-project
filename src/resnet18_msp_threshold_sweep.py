import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data import Subset

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

with torch.no_grad():
    for images, _ in known_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)

        known_confidences.extend(
            confidence.cpu().tolist()
        )

    for images, _ in unknown_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)

        unknown_confidences.extend(
            confidence.cpu().tolist()
        )

thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print()
print("Threshold | Known accepted | Unknown rejected")
print("---------------------------------------------")

for threshold in thresholds:
    known_accepted = sum(
        conf >= threshold for conf in known_confidences
    )

    unknown_rejected = sum(
        conf < threshold for conf in unknown_confidences
    )

    known_rate = 100 * known_accepted / len(known_confidences)
    unknown_rate = 100 * unknown_rejected / len(unknown_confidences)

    print(
        f"{threshold:.1f}       | "
        f"{known_rate:6.2f}%        | "
        f"{unknown_rate:6.2f}%"
    )