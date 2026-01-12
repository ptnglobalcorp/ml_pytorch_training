"""
Exercise 3: Training Loop
PyTorch Workflow Fundamentals - Module 2

This exercise covers:
- Implementing a complete training loop
- Using loss functions
- Setting up optimizers
- Implementing validation
- Saving checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Part 1: Setup - Model, Data, Loss, Optimizer
# ============================================

print("=" * 60)
print("Part 1: Setup")
print("=" * 60)


# Simple model for demonstration
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# TODO: Create dummy data
n_samples = 1000
n_features = 20
n_classes = 5

X_train = torch.randn(n_samples, n_features)
y_train = torch.randint(0, n_classes, (n_samples,))

X_val = torch.randn(200, n_features)
y_val = torch.randint(0, n_classes, (200,))

# TODO: Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# TODO: Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet(n_features, 64, n_classes).to(device)

print(f"Using device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# TODO: Define loss function
criterion = nn.CrossEntropyLoss()

# TODO: Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.001)")


# ============================================
# Part 2: Training Step Function
# ============================================

print("\n" + "=" * 60)
print("Part 2: Training Step")
print("=" * 60)


def train_step(model, dataloader, criterion, optimizer, device):
    """
    Perform one training epoch

    Args:
        model: The neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    # TODO: Implement training loop
    for features, targets in tqdm(dataloader, desc='Training'):
        # Move data to device
        features = features.to(device)
        targets = targets.to(device)

        # TODO: Zero gradients
        optimizer.zero_grad()

        # TODO: Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # TODO: Backward pass
        loss.backward()

        # TODO: Update parameters
        optimizer.step()

        # TODO: Track metrics
        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Calculate average metrics
    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# Test the training step
train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)
print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")


# ============================================
# Part 3: Validation Step Function
# ============================================

print("\n" + "=" * 60)
print("Part 3: Validation Step")
print("=" * 60)


def validate_step(model, dataloader, criterion, device):
    """
    Validate the model

    Args:
        model: The neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Average loss and accuracy
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # TODO: Implement validation loop
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc='Validation'):
            # Move data to device
            features = features.to(device)
            targets = targets.to(device)

            # TODO: Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)

            # TODO: Track metrics
            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Calculate average metrics
    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# Test the validation step
val_loss, val_acc = validate_step(model, val_loader, criterion, device)
print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


# ============================================
# Part 4: Complete Training Loop
# ============================================

print("\n" + "=" * 60)
print("Part 4: Complete Training Loop")
print("=" * 60)


def train_model(model, train_loader, val_loader, criterion, optimizer,
               device, num_epochs=10, save_path='best_model.pth'):
    """
    Complete training loop with validation

    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        save_path: Path to save best model
    """
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        # TODO: Training
        train_loss, train_acc = train_step(
            model, train_loader, criterion, optimizer, device
        )

        # TODO: Validation
        val_loss, val_acc = validate_step(
            model, val_loader, criterion, device
        )

        # TODO: Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # TODO: Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

    return history


# TODO: Run training
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=5,
    save_path='best_model.pth'
)

print("\n" + "=" * 60)
print("Training Complete!")
print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
print("=" * 60)


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Implement learning rate scheduling
print("\nExercise 1: Learning rate scheduling")
# Add a learning rate scheduler that reduces LR when validation loss plateaus

# Exercise 2: Implement gradient clipping
print("\nExercise 2: Gradient clipping")
# Modify train_step to clip gradients before optimizer.step()

# Exercise 3: Implement early stopping
print("\nExercise 3: Early stopping")
# Add early stopping that stops training when validation loss doesn't improve

# Exercise 4: Implement mixed precision training
print("\nExercise 4: Mixed precision training")
# Use torch.cuda.amp for faster training on modern GPUs

# Exercise 5: Implement learning rate warmup
print("\nExercise 5: Learning rate warmup")
# Gradually increase learning rate for the first few epochs


print("\n" + "=" * 60)
print("Exercise 3 Complete!")
print("=" * 60)
