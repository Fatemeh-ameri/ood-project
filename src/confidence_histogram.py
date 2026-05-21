import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# Transform
transform = transforms.ToTensor()

# Dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=False,
    download=False,
    transform=transform
)

# DataLoader
test_loader = DataLoader(
    test_dataset,
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

# Load weights
model.load_state_dict(
    torch.load(
        "models/simple_cnn_cifar10.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

correct_confidences = []
wrong_confidences = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

        correct_mask = predicted == labels
        wrong_mask = predicted != labels

        correct_confidences.extend(
            confidence[correct_mask].cpu().tolist()
        )

        wrong_confidences.extend(
            confidence[wrong_mask].cpu().tolist()
        )

# Plot histogram
plt.figure(figsize=(8, 5))

plt.hist(
    correct_confidences,
    bins=30,
    alpha=0.7,
    label="Correct Predictions"
)

plt.hist(
    wrong_confidences,
    bins=30,
    alpha=0.7,
    label="Wrong Predictions"
)

plt.xlabel("Confidence")
plt.ylabel("Count")

plt.title("Confidence Distribution")

plt.legend()

plt.show()