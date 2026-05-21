import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# CIFAR-10 class names
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

# Transform
transform = transforms.ToTensor()

# Datasets
train_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=True,
    download=False,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=False,
    download=False,
    transform=transform
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True
)

# Model
model = nn.Sequential(
    nn.Flatten(),

    nn.Linear(3 * 32 * 32, 512),
    nn.ReLU(),

    nn.Linear(512, 10)
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 3

for epoch in range(epochs):

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

# Evaluation mode
model.eval()

# Get one batch from test set
images, labels = next(iter(test_loader))

images = images.to(device)

# Predictions
outputs = model(images)

_, predicted = torch.max(outputs, 1)

# Move back to CPU for plotting
images = images.cpu()
predicted = predicted.cpu()

# Plot results
fig, axes = plt.subplots(1, 8, figsize=(15, 3))

for i in range(8):

    image = images[i].permute(1, 2, 0)

    true_label = classes[labels[i]]
    predicted_label = classes[predicted[i]]

    axes[i].imshow(image)

    axes[i].set_title(
        f"T: {true_label}\nP: {predicted_label}",
        fontsize=8
    )

    axes[i].axis("off")

plt.tight_layout()
plt.show()