"""
Exercise 2: Multi-class Classification
Neural Network Classification - Module 3

This exercise covers:
- Building a multi-class classifier
- Using CrossEntropyLoss
- Training and evaluating multi-class classifiers
- Handling multiple classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Part 1: Multi-class Classification Model
# ============================================

print("=" * 60)
print("Part 1: Multi-class Classification Model")
print("=" * 60)


class MultiClassClassifier(nn.Module):
    """Multi-class classification neural network"""

    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassClassifier, self).__init__()

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
            nn.Linear(hidden_size // 2, num_classes)  # One output per class
        )

    def forward(self, x):
        """Forward pass - returns raw logits"""
        return self.network(x)


# TODO: Create the model
num_classes = 5
model = MultiClassClassifier(input_size=20, hidden_size=64, num_classes=num_classes)
print(model)

# TODO: Test with dummy input
dummy_input = torch.randn(10, 20)
output = model(dummy_input)
print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

# TODO: Show prediction probabilities
probs = torch.softmax(output, dim=1)
predictions = torch.argmax(probs, dim=1)
print(f"\nProbabilities (first sample): {probs[0]}")
print(f"Predicted class (first sample): {predictions[0]}")


# ============================================
# Part 2: Loss Function for Multi-class Classification
# ============================================

print("\n" + "=" * 60)
print("Part 2: Loss Function")
print("=" * 60)

# TODO: Use CrossEntropyLoss (recommended)
criterion = nn.CrossEntropyLoss()

# Create sample data
batch_size = 4
num_classes = 5
logits = torch.randn(batch_size, num_classes)
targets = torch.randint(0, num_classes, (batch_size,))

# TODO: Calculate loss
loss = criterion(logits, targets)
print(f"Logits shape: {logits.shape}")
print(f"Targets: {targets}")
print(f"Loss: {loss.item():.4f}")

# TODO: Show manual calculation
probs = torch.softmax(logits, dim=1)
log_probs = torch.log(probs)
nll_loss = nn.NLLLoss()
manual_loss = nll_loss(log_probs, targets)
print(f"\nManual calculation (log_softmax + NLL): {manual_loss.item():.4f}")


# ============================================
# Part 3: Data Preparation
# ============================================

print("\n" + "=" * 60)
print("Part 3: Data Preparation")
print("=" * 60)

# TODO: Generate synthetic multi-class data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=5,  # 5 classes
    n_clusters_per_class=1,
    random_state=42
)

# TODO: Split into train/val/test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# TODO: Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\nClass distribution in training set:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")

# TODO: Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)  # Use LongTensor for class indices
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

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


def train_multiclass_classifier(model, train_loader, val_loader, criterion,
                                 optimizer, device, num_epochs=10):
    """Train multi-class classification model"""

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
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

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
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

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
            torch.save(model.state_dict(), 'best_multiclass_classifier.pth')

    return history


# TODO: Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiClassClassifier(input_size=20, hidden_size=64, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = train_multiclass_classifier(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5
)


# ============================================
# Part 5: Evaluation
# ============================================

print("\n" + "=" * 60)
print("Part 5: Evaluation")
print("=" * 60)


def evaluate_multiclass_classifier(model, test_loader, device, class_names=None):
    """Evaluate multi-class classifier"""

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


# TODO: Load best model and evaluate
model.load_state_dict(torch.load('best_multiclass_classifier.pth'))

class_names = [f'Class {i}' for i in range(5)]
results = evaluate_multiclass_classifier(model, test_loader, device, class_names)

print(f"Test Accuracy: {results['accuracy']:.4f}")
print("\nClassification Report:")
print(classification_report(results['labels'], results['predictions'],
                            target_names=class_names))

print(f"\nConfusion Matrix:")
print(results['confusion_matrix'])


# ============================================
# Part 6: Plot Confusion Matrix
# ============================================

print("\n" + "=" * 60)
print("Part 6: Visualization")
print("=" * 60)


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xlabel='Predicted Label',
        ylabel='True Label',
        title='Confusion Matrix',
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()


plot_confusion_matrix(results['confusion_matrix'], class_names)


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Implement class weighting for imbalanced data
print("\nExercise 1: Class weighting")
# Use sklearn.utils.class_weight.compute_class_weight

# Exercise 2: Implement per-class metrics tracking during training
print("\nExercise 2: Per-class metrics")
# Track precision, recall, F1 for each class during training

# Exercise 3: Implement label smoothing
print("\nExercise 3: Label smoothing")
# Implement LabelSmoothingCrossEntropyLoss

# Exercise 4: Add Top-K accuracy metric
print("\nExercise 4: Top-K accuracy")
# Calculate Top-3 and Top-5 accuracy

# Exercise 5: Implement learning rate scheduling based on accuracy
print("\nExercise 5: Accuracy-based LR scheduling")
# Reduce learning rate when validation accuracy plateaus


print("\n" + "=" * 60)
print("Exercise 2 Complete!")
print("=" * 60)
