import matplotlib.pyplot as plt

# Threshold values
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

# Results from MSP experiment
known_accepted = [84.98, 68.57, 51.60, 37.98, 25.47]

unknown_rejected = [22.68, 45.58, 63.90, 78.08, 88.05]

# Plot
plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    known_accepted,
    marker="o",
    label="Known Accepted"
)

plt.plot(
    thresholds,
    unknown_rejected,
    marker="o",
    label="Unknown Rejected"
)

plt.xlabel("MSP Threshold")
plt.ylabel("Percentage")

plt.title("OOD Threshold Trade-off")

plt.grid(True)

plt.legend()

plt.show()