import os
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("reports/figures", exist_ok=True)

# Thresholds
thresholds_simple_cnn = [0.4, 0.5, 0.6, 0.7, 0.8]
thresholds_resnet18 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Simple CNN results
simple_cnn_known_accepted = [84.98, 68.57, 51.60, 37.98, 25.47]
simple_cnn_unknown_rejected = [22.68, 45.58, 63.90, 78.08, 88.05]

# Improved ResNet18 results
resnet18_known_accepted = [97.08, 91.13, 82.42, 74.03, 65.35, 53.37]
resnet18_unknown_rejected = [9.82, 26.95, 43.45, 57.23, 69.45, 83.03]

# Plot known accepted
plt.figure(figsize=(8, 5))

plt.plot(
    thresholds_simple_cnn,
    simple_cnn_known_accepted,
    marker="o",
    label="Simple CNN - Known Accepted"
)

plt.plot(
    thresholds_resnet18,
    resnet18_known_accepted,
    marker="o",
    label="ResNet18 - Known Accepted"
)

plt.xlabel("MSP Threshold")
plt.ylabel("Percentage")

plt.title("Known Acceptance Across Thresholds")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/figures/known_acceptance_threshold_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Plot unknown rejected
plt.figure(figsize=(8, 5))

plt.plot(
    thresholds_simple_cnn,
    simple_cnn_unknown_rejected,
    marker="o",
    label="Simple CNN - Unknown Rejected"
)

plt.plot(
    thresholds_resnet18,
    resnet18_unknown_rejected,
    marker="o",
    label="ResNet18 - Unknown Rejected"
)

plt.xlabel("MSP Threshold")
plt.ylabel("Percentage")

plt.title("Unknown Rejection Across Thresholds")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/figures/unknown_rejection_threshold_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()