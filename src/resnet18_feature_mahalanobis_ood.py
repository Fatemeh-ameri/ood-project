import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.metrics import roc_auc_score, roc_curve

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

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

known_class_ids = [
    classes.index(name)
    for name in known_class_names
]

unknown_class_ids = [
    classes.index(name)
    for name in unknown_class_names
]

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

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
    i for i, (_, label)
    in enumerate(train_dataset)
    if label in known_class_ids
]

known_test_indices = [
    i for i, (_, label)
    in enumerate(test_dataset)
    if label in known_class_ids
]

unknown_test_indices = [
    i for i, (_, label)
    in enumerate(test_dataset)
    if label in unknown_class_ids
]

known_train_dataset = Subset(
    train_dataset,
    known_train_indices
)

known_test_dataset = Subset(
    test_dataset,
    known_test_indices
)

unknown_test_dataset = Subset(
    test_dataset,
    unknown_test_indices
)

known_train_loader = DataLoader(
    known_train_dataset,
    batch_size=64,
    shuffle=False
)

known_test_loader = DataLoader(
    known_test_dataset,
    batch_size=64,
    shuffle=False
)

unknown_test_loader = DataLoader(
    unknown_test_dataset,
    batch_size=64,
    shuffle=False
)

# Load ResNet18
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model.load_state_dict(
    torch.load(
        "models/known_only_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Known-only ResNet18 loaded!")

# Feature extractor: all layers except final fc
feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


def extract_features(images):
    features = feature_extractor(images)

    # ResNet output shape: [batch, 512, 1, 1]
    features = torch.flatten(features, start_dim=1)

    return features


# -----------------------------
# Step 1: Extract known train features
# -----------------------------

train_features = []
train_labels = []

with torch.no_grad():
    for images, labels in known_train_loader:
        images = images.to(device)

        features = extract_features(images)

        train_features.append(features.cpu())
        train_labels.extend(labels.tolist())

train_features = torch.cat(train_features, dim=0)
train_labels = torch.tensor(train_labels)

print("Known train features extracted:", train_features.shape)


# -----------------------------
# Step 2: Compute class means
# -----------------------------

class_means = {}

for class_id in known_class_ids:
    class_features = train_features[
        train_labels == class_id
    ]

    class_means[class_id] = class_features.mean(dim=0)

print("Class means computed.")


# -----------------------------
# Step 3: Compute shared covariance
# -----------------------------

centered_features = []

for feature, label in zip(train_features, train_labels):
    mean = class_means[label.item()]
    centered_features.append(feature - mean)

centered_features = torch.stack(centered_features)

# Covariance matrix: [512, 512]
covariance = torch.cov(centered_features.T)

# Add small regularization for numerical stability
epsilon = 1e-3
identity = torch.eye(covariance.size(0))

regularized_covariance = covariance + epsilon * identity

inverse_covariance = torch.linalg.pinv(regularized_covariance)

print("Shared covariance inverse computed.")


# -----------------------------
# Step 4: Mahalanobis score
# -----------------------------

def mahalanobis_score(features):
    means = torch.stack(
        list(class_means.values())
    )

    features = features.cpu()

    distances = []

    for mean in means:
        diff = features - mean

        # Mahalanobis distance:
        # (x - mean)^T Sigma^-1 (x - mean)
        distance = torch.sum(
            (diff @ inverse_covariance) * diff,
            dim=1
        )

        distances.append(distance)

    distances = torch.stack(distances, dim=1)

    nearest_distance = distances.min(dim=1).values

    # Lower distance means more likely known.
    # AUROC expects higher score for positive class.
    score = -nearest_distance

    return score


scores = []
binary_labels = []

# Known test samples: label = 1
with torch.no_grad():
    for images, _ in known_test_loader:
        images = images.to(device)

        features = extract_features(images)

        known_scores = mahalanobis_score(features)

        scores.extend(known_scores.tolist())
        binary_labels.extend([1] * images.size(0))

# Unknown test samples: label = 0
with torch.no_grad():
    for images, _ in unknown_test_loader:
        images = images.to(device)

        features = extract_features(images)

        unknown_scores = mahalanobis_score(features)

        scores.extend(unknown_scores.tolist())
        binary_labels.extend([0] * images.size(0))


# -----------------------------
# Step 5: AUROC and FPR@95TPR
# -----------------------------

auroc = roc_auc_score(
    binary_labels,
    scores
)

print(f"Mahalanobis feature AUROC: {auroc:.4f}")


def calculate_fpr_at_95_tpr(labels, scores):
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    target_tpr = 0.95

    valid_indices = [
        index for index, value in enumerate(tpr)
        if value >= target_tpr
    ]

    if not valid_indices:
        raise ValueError("No threshold reaches 95% TPR.")

    best_index = valid_indices[0]

    return fpr[best_index], tpr[best_index], thresholds[best_index]


fpr95, tpr, threshold = calculate_fpr_at_95_tpr(
    binary_labels,
    scores
)

print(
    f"Mahalanobis feature FPR@95TPR: "
    f"FPR={fpr95:.4f}, "
    f"TPR={tpr:.4f}, "
    f"Threshold={threshold:.4f}"
)