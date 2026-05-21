import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Select device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# Transform images into tensors
transform = transforms.ToTensor()

# Download CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=True,
    download=True,
    transform=transform
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Get one batch
images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

print("First 10 labels:")
print(labels[:10])