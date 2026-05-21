import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

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

known_class_names = [
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse"
]

unknown_class_names = [
    "airplane",
    "automobile",
    "ship",
    "truck"
]

known_class_ids = [classes.index(name) for name in known_class_names]
unknown_class_ids = [classes.index(name) for name in unknown_class_names]

transform = transforms.ToTensor()

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

known_train_indices = [
    i for i, (_, label) in enumerate(train_dataset)
    if label in known_class_ids
]

known_test_indices = [
    i for i, (_, label) in enumerate(test_dataset)
    if label in known_class_ids
]

unknown_test_indices = [
    i for i, (_, label) in enumerate(test_dataset)
    if label in unknown_class_ids
]

known_train_dataset = Subset(train_dataset, known_train_indices)
known_test_dataset = Subset(test_dataset, known_test_indices)
unknown_test_dataset = Subset(test_dataset, unknown_test_indices)

print("Known class IDs:", known_class_ids)
print("Unknown class IDs:", unknown_class_ids)

print("Known train samples:", len(known_train_dataset))
print("Known test samples:", len(known_test_dataset))
print("Unknown test samples:", len(unknown_test_dataset))