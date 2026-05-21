import torch
import torch.nn as nn
import torch.optim as optim

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

# Known classes only
known_class_names = [
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse"
]

known_class_ids = [
    classes.index(name)
    for name in known_class_names
]

print("Known classes:", known_class_names)

# Transform with normalization
transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# Training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=True,
    download=False,
    transform=transform
)

# Keep only known-class samples
known_train_indices = [
    i for i, (_, label)
    in enumerate(train_dataset)
    if label in known_class_ids
]

known_train_dataset = Subset(
    train_dataset,
    known_train_indices
)

print("Known training samples:", len(known_train_dataset))

# DataLoader
train_loader = DataLoader(
    known_train_dataset,
    batch_size=64,
    shuffle=True
)

# ResNet18
model = models.resnet18(weights=None)

# Replace final layer for CIFAR-10
model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model = model.to(device)

print("Known-only ResNet18 loaded!")

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )

# Save model
torch.save(
    model.state_dict(),
    "models/known_only_resnet18.pth"
)

print("Known-only ResNet18 model saved!")