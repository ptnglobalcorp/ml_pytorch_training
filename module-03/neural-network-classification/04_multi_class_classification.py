"""
Exercise 4: Multi-Class Classification
Neural Network Classification - Module 3

This exercise covers:
- Creating multi-class data with make_blobs
- Building multi-class models
- Using CrossEntropyLoss
- Converting logits with softmax
- Using argmax for predictions

Learning Mottos:
- If in doubt, run the code!
- Experiment, experiment, experiment!
- Visualize, visualize, visualize!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Part 1: Creating make_blobs Dataset
# ============================================

print("=" * 60)
print("Part 1: Creating make_blobs Dataset")
print("=" * 60)

# TODO: Create make_blobs dataset
n_samples = 1000
n_classes = 4

X, y = make_blobs(
    n_samples=n_samples,
    n_features=2,
    centers=n_classes,
    cluster_std=1.5,
    random_state=42
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of classes: {len(set(y))}")

for class_id in range(n_classes):
    print(f"  Class {class_id}: {sum(y == class_id)} samples")

# ============================================
# Part 2: Visualizing the Data
# ============================================

print("\n" + "=" * 60)
print("Part 2: Visualizing the Data")
print("=" * 60)

# TODO: Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=10, alpha=0.6)
plt.title(f'make_blobs Multi-class Dataset ({n_classes} classes)', fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Class')
plt.tight_layout()
plt.show()

print("Visualization created!")
print("Motto: Visualize, visualize, visualize!")

# ============================================
# Part 3: Train/Test Split and Conversion
# ============================================

print("\n" + "=" * 60)
print("Part 3: Train/Test Split and Conversion")
print("=" * 60)

# TODO: Split data (stratified for multi-class)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# TODO: Convert to PyTorch tensors
# IMPORTANT: For multi-class, y should be LongTensor (class indices)
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Move to device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print(f"\nTensor shapes:")
print(f"X_train: {X_train.shape}, dtype: {X_train.dtype}")
print(f"y_train: {y_train.shape}, dtype: {y_train.dtype}")
print(f"X_test: {X_test.shape}, dtype: {X_test.dtype}")
print(f"y_test: {y_test.shape}, dtype: {y_test.dtype}")

# ============================================
# Part 4: Building Multi-Class Model
# ============================================

print("\n" + "=" * 60)
print("Part 4: Building Multi-Class Model")
print("=" * 60)

# TODO: Define multi-class model
class BlobModel(nn.Module):
    """Multi-class classifier"""
    def __init__(self, input_size=2, hidden_size=16, num_classes=4):
        super(BlobModel, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
            # Note: No softmax here (output is logits)
        )

    def forward(self, x):
        return self.layer_stack(x)

model = BlobModel(input_size=2, hidden_size=16, num_classes=n_classes)
model = model.to(device)

print("BlobModel (Multi-class):")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ============================================
# Part 5: Loss Function and Optimizer
# ============================================

print("\n" + "=" * 60)
print("Part 5: Loss Function and Optimizer")
print("=" * 60)

# TODO: Use CrossEntropyLoss for multi-class
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"Loss function: CrossEntropyLoss()")
print(f"Optimizer: Adam (lr=0.01)")

print("\nKey differences from binary:")
print("  - Binary: BCEWithLogitsLoss, y is FloatTensor with shape [N, 1]")
print("  - Multi-class: CrossEntropyLoss, y is LongTensor with shape [N]")

# ============================================
# Part 6: Training Loop
# ============================================

print("\n" + "=" * 60)
print("Part 6: Training Loop")
print("=" * 60)

epochs = 100
train_losses = []
train_accuracies = []

model.train()
for epoch in range(epochs):
    # Forward pass
    y_logits = model(X_train)

    # Calculate loss
    loss = criterion(y_logits, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Track metrics
    train_losses.append(loss.item())
    with torch.no_grad():
        y_pred = torch.argmax(y_logits, dim=1)
        accuracy = (y_pred == y_train).float().mean()
        train_accuracies.append(accuracy.item())

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Acc: {accuracy.item()*100:.2f}%')

print("\nTraining complete!")

# ============================================
# Part 7: The Logits → Softmax → Argmax Pipeline
# ============================================

print("\n" + "=" * 60)
print("Part 7: Logits → Probabilities → Labels Pipeline")
print("=" * 60)

# Make predictions
model.eval()
with torch.inference_mode():
    test_logits = model(X_test)

    # Convert to probabilities using softmax
    test_probs = torch.softmax(test_logits, dim=1)

    # Convert to labels using argmax
    test_preds = torch.argmax(test_probs, dim=1)

# Show examples
print("\nFirst 3 predictions:")
for i in range(3):
    print(f"\nSample {i+1}:")
    print(f"  Logits: {test_logits[i].cpu().numpy()}")
    print(f"  Probabilities: {test_probs[i].cpu().numpy()}")
    print(f"  Predicted: {test_preds[i].item()}, Actual: {y_test[i].item()}")

# Calculate test accuracy
test_accuracy = (test_preds == y_test).float().mean()
print(f"\nTest Accuracy: {test_accuracy.item()*100:.2f}%")

# ============================================
# Part 8: Decision Boundary Visualization
# ============================================

print("\n" + "=" * 60)
print("Part 8: Decision Boundary Visualization")
print("=" * 60)

def plot_multi_class_decision_boundary(model, X, y, title):
    """Plot decision boundary for multi-class model"""
    model.eval()

    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Make predictions on meshgrid
    mesh = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    with torch.inference_mode():
        Z = torch.argmax(model(mesh), dim=1).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z.cpu().numpy(), alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='RdYlBu', edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Class')
    plt.tight_layout()
    plt.show()

# Plot decision boundary
X_test_np = X_test.cpu().numpy()
y_test_np = y_test.cpu().numpy()

plot_multi_class_decision_boundary(
    model, X_test_np, y_test_np,
    f'Multi-class Decision Boundary - Accuracy: {test_accuracy.item()*100:.1f}%'
)

# ============================================
# Part 9: Per-Class Accuracy
# ============================================

print("\n" + "=" * 60)
print("Part 9: Per-Class Accuracy")
print("=" * 60)

for class_id in range(n_classes):
    class_mask = y_test == class_id
    class_acc = (test_preds[class_mask] == y_test[class_mask]).float().mean()
    print(f"Class {class_id} Accuracy: {class_acc.item()*100:.2f}%")

# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Vary number of classes
print("\nExercise 1: Vary number of classes")
print("Try different numbers of classes:")
print("  - n_classes=2 (binary)")
print("  - n_classes=3")
print("  - n_classes=4 (default)")
print("  - n_classes=5")
print("Tip: Modify the 'n_classes' and 'centers' parameters")

# Exercise 2: Change cluster standard deviation
print("\nExercise 2: Change cluster overlap")
print("Try different cluster_std values:")
print("  - cluster_std=0.5 (tight clusters)")
print("  - cluster_std=1.5 (default)")
print("  - cluster_std=3.0 (overlapping clusters)")
print("Tip: Modify the 'cluster_std' parameter in make_blobs()")

# Exercise 3: Compare binary vs multi-class
print("\nExercise 3: Compare binary vs multi-class")
print("Questions to answer:")
print("  - How is binary different from multi-class?")
print("  - What's different about the loss function?")
print("  - What's different about the output activation?")
print("Tip: Look at BCEWithLogitsLoss vs CrossEntropyLoss")

# Exercise 4: Experiment with model size
print("\nExercise 4: Experiment with model size")
print("Try different model sizes:")
print("  - hidden_size=8")
print("  - hidden_size=16 (default)")
print("  - hidden_size=32")
print("Tip: Modify the 'hidden_size' parameter in BlobModel()")

# Exercise 5: Add more hidden layers
print("\nExercise 5: Add more hidden layers")
print("Try deeper architectures:")
print("  - Add another hidden layer")
print("  - Experiment with different widths")
print("Tip: Modify BlobModel to add more layers")

print("\n" + "=" * 60)
print("Exercise 4 Complete!")
print("Remember: If in doubt, run the code!")
print("Remember: Experiment, experiment, experiment!")
print("Remember: Visualize, visualize, visualize!")
print("=" * 60)
