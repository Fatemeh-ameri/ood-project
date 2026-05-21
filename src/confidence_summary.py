import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

transform = transforms.ToTensor()

test_dataset = torchvision.datasets.CIFAR10(
    root="./data/raw",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

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

        correct_confidences.extend(confidence[correct_mask].cpu().tolist())
        wrong_confidences.extend(confidence[wrong_mask].cpu().tolist())

avg_correct_conf = sum(correct_confidences) / len(correct_confidences)
avg_wrong_conf = sum(wrong_confidences) / len(wrong_confidences)

print(f"Number of correct predictions: {len(correct_confidences)}")
print(f"Number of wrong predictions: {len(wrong_confidences)}")
print(f"Average confidence when correct: {avg_correct_conf:.3f}")
print(f"Average confidence when wrong:   {avg_wrong_conf:.3f}")