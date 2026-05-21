import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

# Dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=True,
    download=False,
    transform=transform
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

# Get one batch
images, labels = next(iter(train_loader))

# Plot images
fig, axes = plt.subplots(1, 8, figsize=(15, 3))

for i in range(8):
    image = images[i].permute(1, 2, 0)
    
    axes[i].imshow(image)
    axes[i].set_title(classes[labels[i]])
    axes[i].axis("off")

plt.tight_layout()
plt.show()