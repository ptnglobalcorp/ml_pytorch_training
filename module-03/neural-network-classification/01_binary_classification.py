"""
Exercise 1: Binary Classification
Neural Network Classification - Module 3

This exercise covers:
- Building a binary classifier
- Using appropriate loss functions
- Training and evaluating binary classifiers
- Handling class imbalance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Part 1: Binary Classification Model
# ============================================

print("=" * 60)
print("Part 1: Binary Classification Model")
print("=" * 60)


class BinaryClassifier(nn.Module):
    """Binary classification neural network"""

    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()

        # TODO: Define the network layers
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)  # Single output for binary classification
        )

    def forward(self, x):
        """Forward pass - returns raw logits"""
        return self.network(x)


# TODO: Create the model
model = BinaryClassifier(input_size=20, hidden_size=64)
print(model)

# TODO: Test with dummy input
dummy_input = torch.randn(10, 20)
output = model(dummy_input)
print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output (logits): {output.squeeze()[:3]}")


# ============================================
# Part 2: Loss Function for Binary Classification
# ============================================

print("\n" + "=" * 60)
print("Part 2: Loss Function")
print("=" * 60)

# TODO: Use BCEWithLogitsLoss (recommended)
criterion = nn.BCEWithLogitsLoss()

# Create sample data
logits = torch.randn(5)
targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])

# TODO: Calculate loss
loss = criterion(logits, targets)
print(f"Logits: {logits}")
print(f"Targets: {targets}")
print(f"Loss: {loss.item():.4f}")

# TODO: Show sigmoid + BCELoss equivalent
sigmoid_logits = torch.sigmoid(logits)
criterion_bce = nn.BCELoss()
loss_bce = criterion_bce(sigmoid_logits, targets)
print(f"\nUsing Sigmoid + BCELoss: {loss_bce.item():.4f}")
print(f"Using BCEWithLogitsLoss: {loss.item():.4f}")


# ============================================
# Part 3: Data Preparation
# ============================================

print("\n" + "=" * 60)
print("Part 3: Data Preparation")
print("=" * 60)

# TODO: Generate synthetic binary classification data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# TODO: Split into train/val/test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# TODO: Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# TODO: Create dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ============================================
# Part 4: Training Function
# ============================================

print("\n" + "=" * 60)
print("Part 4: Training Function")
print("=" * 60)


def train_binary_classifier(model, train_loader, val_loader, criterion,
                             optimizer, device, num_epochs=10):
    """Train binary classification model"""

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += preds.eq(y_batch.long()).sum().item()
            total += y_batch.size(0)

        train_loss /= total
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct += preds.eq(y_batch.long()).sum().item()
                total += y_batch.size(0)

        val_loss /= total
        val_acc = 100. * correct / total

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_binary_classifier.pth')

    return history


# TODO: Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BinaryClassifier(input_size=20, hidden_size=64).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = train_binary_classifier(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5
)


# ============================================
# Part 5: Evaluation
# ============================================

print("\n" + "=" * 60)
print("Part 5: Evaluation")
print("=" * 60)


def evaluate_binary_classifier(model, test_loader, device):
    """Evaluate binary classifier and return metrics"""

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs
    }


# TODO: Load best model and evaluate
model.load_state_dict(torch.load('best_binary_classifier.pth'))
results = evaluate_binary_classifier(model, test_loader, device)

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test Precision: {results['precision']:.4f}")
print(f"Test Recall: {results['recall']:.4f}")
print(f"Test F1 Score: {results['f1']:.4f}")
print(f"\nConfusion Matrix:")
print(results['confusion_matrix'])


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Implement weighted loss for class imbalance
print("\nExercise 1: Weighted loss for class imbalance")
# Calculate class weights and use them in BCEWithLogitsLoss

# Exercise 2: Implement ROC curve and AUC calculation
print("\nExercise 2: ROC curve and AUC")
# Use sklearn.metrics to calculate ROC curve and AUC

# Exercise 3: Implement threshold tuning
print("\nExercise 3: Threshold tuning")
# Find optimal threshold different from 0.5 using validation set

# Exercise 4: Add early stopping to training
print("\nExercise 4: Early stopping")
# Stop training when validation loss doesn't improve

# Exercise 5: Implement focal loss for extreme class imbalance
print("\nExercise 5: Focal loss")
# Implement focal loss for handling extreme class imbalance


print("\n" + "=" * 60)
print("Exercise 1 Complete!")
print("=" * 60)
