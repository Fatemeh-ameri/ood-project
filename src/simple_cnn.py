import torch
import torch.nn as nn

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Simple CNN model
model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),

    nn.Linear(32 * 8 * 8, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)

model = model.to(device)

print(model)

# Fake batch for testing
x = torch.randn(64, 3, 32, 32).to(device)

outputs = model(x)

print("Output shape:", outputs.shape)