import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.metrics import roc_auc_score, roc_curve


# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# CIFAR-10 classes
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

# Evaluation transform
transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

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

# Known train subset
known_train_indices = [
    i for i, (_, label)
    in enumerate(train_dataset)
    if label in known_class_ids
]

# Known test subset
known_test_indices = [
    i for i, (_, label)
    in enumerate(test_dataset)
    if label in known_class_ids
]

# Unknown test subset
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

# ResNet18 model
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

# Feature extractor:
# everything except the final classification layer
feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


def extract_features(images):
    features = feature_extractor(images)

    features = torch.flatten(features, start_dim=1)

    # Normalize features so that similarity is based on direction,
    # not feature vector magnitude.
    features = F.normalize(features, p=2, dim=1)

    return features


# -----------------------------
# Step 1: Build class centers
# -----------------------------

features_by_class = {
    class_id: []
    for class_id in known_class_ids
}

with torch.no_grad():

    for images, labels in known_train_loader:

        images = images.to(device)

        features = extract_features(images)

        features = features.cpu()
        labels = labels.cpu()

        for feature, label in zip(features, labels):

            if label.item() in known_class_ids:
                features_by_class[label.item()].append(feature)

class_centers = {}

for class_id, features in features_by_class.items():

    stacked_features = torch.stack(features)

    center = stacked_features.mean(dim=0)

    class_centers[class_id] = center

print("Class centers created for known classes.")


# -----------------------------
# Step 2: Score test samples
# -----------------------------

def max_center_similarity(features):
    centers = torch.stack(
        list(class_centers.values())
    ).to(features.device)

    # Normalize centers too, for cosine similarity.
    centers = F.normalize(centers, p=2, dim=1)

    # features shape: [batch, 512]
    # centers shape: [num_known_classes, 512]
    # similarities shape: [batch, num_known_classes]
    similarities = features @ centers.T

    max_similarities = similarities.max(dim=1).values

    return max_similarities


scores = []
binary_labels = []

# Known test samples: label = 1
with torch.no_grad():

    for images, _ in known_test_loader:

        images = images.to(device)

        features = extract_features(images)

        known_scores = max_center_similarity(features)

        scores.extend(
            known_scores.cpu().tolist()
        )

        binary_labels.extend(
            [1] * images.size(0)
        )

# Unknown test samples: label = 0
with torch.no_grad():

    for images, _ in unknown_test_loader:

        images = images.to(device)

        features = extract_features(images)

        unknown_scores = max_center_similarity(features)

        scores.extend(
            unknown_scores.cpu().tolist()
        )

        binary_labels.extend(
            [0] * images.size(0)
        )

auroc = roc_auc_score(
    binary_labels,
    scores
)

print(f"Feature cosine-center AUROC: {auroc:.4f}")


# -----------------------------
# Step 3: FPR@95TPR
# -----------------------------

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
    f"Feature cosine-center FPR@95TPR: "
    f"FPR={fpr95:.4f}, "
    f"TPR={tpr:.4f}, "
    f"Threshold={threshold:.4f}"
)