import torch
import torch.nn as nn

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Simple neural network
model = nn.Sequential(
    nn.Flatten(),

    nn.Linear(3 * 32 * 32, 512),
    nn.ReLU(),

    nn.Linear(512, 10)
)

# Move model to device
model = model.to(device)

print(model)

# Fake batch of images
x = torch.randn(64, 3, 32, 32).to(device)

# Forward pass
outputs = model(x)

print("Output shape:", outputs.shape)