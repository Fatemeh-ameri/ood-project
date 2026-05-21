import torch

# Select device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0], device=device)
y = torch.tensor([4.0, 5.0, 6.0], device=device)

# Tensor operation
z = x + y

print("x:", x)
print("y:", y)
print("z:", z)