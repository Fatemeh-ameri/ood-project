import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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

# Threshold
threshold = 0.60

print(f"MSP Threshold: {threshold}")

# Transform
transform = transforms.ToTensor()

# Dataset
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

# -----------------------------
# Known detection
# -----------------------------

known_detected_as_known = 0
known_total = 0

with torch.no_grad():

    for images, _ in known_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        known_detected_as_known += (
            confidence >= threshold
        ).sum().item()

        known_total += images.size(0)

# -----------------------------
# Unknown detection
# -----------------------------

unknown_detected_as_unknown = 0
unknown_total = 0

with torch.no_grad():

    for images, _ in unknown_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        unknown_detected_as_unknown += (
            confidence < threshold
        ).sum().item()

        unknown_total += images.size(0)

# Results
known_detection_rate = (
    100 * known_detected_as_known / known_total
)

unknown_detection_rate = (
    100 * unknown_detected_as_unknown / unknown_total
)

print()
print(f"Known detected as KNOWN: {known_detection_rate:.2f}%")
print(f"Unknown detected as UNKNOWN: {unknown_detection_rate:.2f}%")