import os

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

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

known_indices = [
    i for i, (_, label)
    in enumerate(test_dataset)
    if label in known_class_ids
]

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

model.load_state_dict(
    torch.load(
        "models/known_only_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Known-only ResNet18 loaded!")

all_logits = []
all_labels = []

# Known samples: label = 1
with torch.no_grad():
    for images, _ in known_loader:
        images = images.to(device)

        logits = model(images)

        all_logits.append(logits.cpu())
        all_labels.extend([1] * images.size(0))

# Unknown samples: label = 0
with torch.no_grad():
    for images, _ in unknown_loader:
        images = images.to(device)

        logits = model(images)

        all_logits.append(logits.cpu())
        all_labels.extend([0] * images.size(0))

all_logits = torch.cat(all_logits, dim=0)

# MSP score
probabilities = torch.softmax(all_logits, dim=1)

msp_scores = torch.max(
    probabilities,
    dim=1
).values

# Energy score with best temperature found earlier
temperature = 2

energy_scores = temperature * torch.logsumexp(
    all_logits / temperature,
    dim=1
)

# AUROC values
msp_auroc = roc_auc_score(
    all_labels,
    msp_scores.tolist()
)

energy_auroc = roc_auc_score(
    all_labels,
    energy_scores.tolist()
)

print(f"MSP AUROC: {msp_auroc:.4f}")
print(f"Energy AUROC: {energy_auroc:.4f}")

# ROC curves
msp_fpr, msp_tpr, _ = roc_curve(
    all_labels,
    msp_scores.tolist()
)

energy_fpr, energy_tpr, _ = roc_curve(
    all_labels,
    energy_scores.tolist()
)

# Save plot
os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(6, 5))

plt.plot(
    msp_fpr,
    msp_tpr,
    label=f"MSP AUROC = {msp_auroc:.4f}"
)

plt.plot(
    energy_fpr,
    energy_tpr,
    label=f"Energy AUROC = {energy_auroc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("MSP vs Energy ROC Curve")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reports/figures/msp_energy_roc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()