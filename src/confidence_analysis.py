import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

# Test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=False,
    download=False,
    transform=transform
)

# DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True
)

# Recreate CNN model
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

# Load saved weights
model.load_state_dict(
    torch.load(
        "models/simple_cnn_cifar10.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Get one batch
images, labels = next(iter(test_loader))

images = images.to(device)

with torch.no_grad():
    outputs = model(images)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)

# Print results
for i in range(len(images)):
    true_label = classes[labels[i]]
    predicted_label = classes[predicted[i].cpu()]
    conf = confidence[i].item()

    print(
        f"True: {true_label:10s} | "
        f"Predicted: {predicted_label:10s} | "
        f"Confidence: {conf:.2f}"
    )