import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.metrics import roc_auc_score

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

# Same transform used for ResNet18 evaluation
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

# Load improved known-only ResNet18
model.load_state_dict(
    torch.load(
        "models/known_only_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Known-only ResNet18 loaded!")

scores = []
labels = []

# Known samples: label = 1
with torch.no_grad():
    for images, _ in known_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        scores.extend(confidence.cpu().tolist())
        labels.extend([1] * images.size(0))

# Unknown samples: label = 0
with torch.no_grad():
    for images, _ in unknown_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        scores.extend(confidence.cpu().tolist())
        labels.extend([0] * images.size(0))

auroc = roc_auc_score(labels, scores)

print(f"MSP AUROC: {auroc:.4f}")