import matplotlib.pyplot as plt

# Thresholds
thresholds_simple_cnn = [0.4, 0.5, 0.6, 0.7, 0.8]
thresholds_resnet18 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Simple CNN results
simple_cnn_known_accepted = [84.98, 68.57, 51.60, 37.98, 25.47]
simple_cnn_unknown_rejected = [22.68, 45.58, 63.90, 78.08, 88.05]

# ResNet18 results
resnet18_known_accepted = [98.38, 94.33, 86.78, 79.50, 71.37, 61.08]
resnet18_unknown_rejected = [3.83, 14.38, 26.75, 38.98, 51.88, 66.20]

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
plt.show()