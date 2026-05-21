import torch
import torch.nn as nn
import torchvision.models as models

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# Load ResNet18
model = models.resnet18(weights=None)

# Change final layer for CIFAR-10
model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model = model.to(device)

print(model.fc)

# Fake batch
x = torch.randn(64, 3, 32, 32).to(device)

# Forward pass
outputs = model(x)

print("Output shape:", outputs.shape)