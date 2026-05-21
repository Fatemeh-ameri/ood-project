import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data import Subset

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

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

known_class_names = [
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse"
]

unknown_class_names = [
    "airplane",
    "automobile",
    "ship",
    "truck"
]

known_class_ids = [classes.index(name) for name in known_class_names]
unknown_class_ids = [classes.index(name) for name in unknown_class_names]

transform = transforms.ToTensor()

test_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=False,
    download=False,
    transform=transform
)

known_indices = [
    i for i, (_, label) in enumerate(test_dataset)
    if label in known_class_ids
]

unknown_indices = [
    i for i, (_, label) in enumerate(test_dataset)
    if label in unknown_class_ids
]

known_dataset = Subset(test_dataset, known_indices)
unknown_dataset = Subset(test_dataset, unknown_indices)

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

model.load_state_dict(
    torch.load(
        "models/known_only_cnn.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

known_confidences = []
unknown_confidences = []

with torch.no_grad():
    for images, _ in known_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)

        known_confidences.extend(confidence.cpu().tolist())

    for images, _ in unknown_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)

        unknown_confidences.extend(confidence.cpu().tolist())

thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

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