# Classification Basics

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the difference between binary and multi-class classification
- Prepare classification datasets
- Choose appropriate loss functions for classification
- Implement proper output layers and activation functions
- Handle class imbalance in classification tasks

## Introduction to Classification

Classification is the task of predicting discrete class labels from input features. There are three main types:

| Type | Description | Number of Classes | Example |
|------|-------------|-------------------|---------|
| **Binary Classification** | Two mutually exclusive classes | 2 | Spam detection |
| **Multi-class Classification** | More than two mutually exclusive classes | 3+ | Digit recognition (0-9) |
| **Multi-label Classification** | Multiple labels can be active | N (independent) | Tag classification |

## Binary Classification

### Problem Setup

Binary classification predicts between two classes (typically labeled 0 and 1, or negative/positive).

### Model Architecture

```python
import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)  # Single output unit
        )

    def forward(self, x):
        return self.network(x)
```

### Loss Function and Output

For binary classification, we have two approaches:

#### Approach 1: Sigmoid + BCELoss

```python
# Model with sigmoid activation
class BinaryClassifierWithSigmoid(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output probability in [0, 1]
        )

    def forward(self, x):
        return self.network(x)

# Loss function
criterion = nn.BCELoss()

# Training
model = BinaryClassifierWithSigmoid(input_size=20, hidden_size=64)
logits = model(X_batch)  # Probability in [0, 1]
loss = criterion(logits, y_batch.float())
```

#### Approach 2: Logits + BCEWithLogitsLoss (Recommended)

```python
# Model without sigmoid (outputs raw logits)
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Raw logits (no sigmoid)
        )

    def forward(self, x):
        return self.network(x)

# Loss function (more numerically stable)
criterion = nn.BCEWithLogitsLoss()

# Training
model = BinaryClassifier(input_size=20, hidden_size=64)
logits = model(X_batch)  # Raw logits
loss = criterion(logits, y_batch.float())

# For predictions
probabilities = torch.sigmoid(logits)
predictions = (probabilities > 0.5).long()
```

> **Note**: `BCEWithLogitsLoss` is more numerically stable because it combines sigmoid and binary cross-entropy in a single operation.

### Complete Binary Classification Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Generate synthetic binary classification data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Create datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = BinaryClassifier(input_size=20, hidden_size=64)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss/len(train_loader):.4f}')
    print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print('-' * 60)
```

## Multi-class Classification

### Problem Setup

Multi-class classification predicts one class from three or more mutually exclusive classes.

### Model Architecture

```python
class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)  # One output per class
        )

    def forward(self, x):
        return self.network(x)
```

### Loss Function and Output

For multi-class classification, we use:

```python
# Loss function: CrossEntropyLoss
# Note: This combines LogSoftmax and NLLLoss in one class
criterion = nn.CrossEntropyLoss()

# Model outputs raw logits (no softmax in forward pass)
model = MultiClassClassifier(input_size=20, hidden_size=64, num_classes=5)

# Training
logits = model(X_batch)  # Shape: (batch_size, num_classes)
loss = criterion(logits, y_batch.long())  # y_batch should be class indices

# For predictions
probabilities = torch.softmax(logits, dim=1)
predictions = torch.argmax(probabilities, dim=1)
```

> **Important**: `CrossEntropyLoss` expects raw logits (no softmax) and class indices as targets (not one-hot encoded).

### Complete Multi-class Classification Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic multi-class data
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=5,  # 5 classes
    n_clusters_per_class=1,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)  # Use LongTensor for class indices
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Create datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = MultiClassClassifier(input_size=20, hidden_size=64, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 15
for epoch in range(num_epochs):
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

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    train_accuracy = 100. * correct / total

    # Evaluation
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

    test_accuracy = accuracy_score(all_labels, all_preds)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
    print(f'Test Acc: {test_accuracy:.4f}')
    print('-' * 60)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
```

## Multi-label Classification

### Problem Setup

Multi-label classification predicts multiple labels simultaneously. Each label is independent.

### Model Architecture and Loss

```python
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(MultiLabelClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels)  # One output per label
        )

    def forward(self, x):
        return self.network(x)

# Loss function: BCEWithLogitsLoss for each label independently
model = MultiLabelClassifier(input_size=20, hidden_size=64, num_labels=5)
criterion = nn.BCEWithLogitsLoss()

# Training
logits = model(X_batch)  # Shape: (batch_size, num_labels)
loss = criterion(logits, y_batch.float())  # y_batch is multi-hot encoded

# For predictions
probabilities = torch.sigmoid(logits)
predictions = (probabilities > 0.5).long()
```

## Handling Class Imbalance

Class imbalance is common in real-world classification tasks. Here are strategies to handle it:

### 1. Weighted Loss Function

```python
# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Use weighted loss
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

### 2. Oversampling/Undersampling

```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train.numpy(), y_train.numpy())

# Undersample majority classes
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train.numpy(), y_train.numpy())
```

### 3. Focal Loss (for extreme imbalance)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Usage
criterion = FocalLoss(alpha=1, gamma=2)
```

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# For binary classification
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# For multi-class classification
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Detailed report
report = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(report)
```

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Binary Classification** | Two classes, use BCEWithLogitsLoss |
| **Multi-class Classification** | 3+ mutually exclusive classes, use CrossEntropyLoss |
| **Multi-label Classification** | Multiple independent labels, use BCEWithLogitsLoss |
| **Class Imbalance** | Handle with weighted loss, resampling, or focal loss |
| **Evaluation Metrics** | Accuracy, precision, recall, F1-score, confusion matrix |

## Practice Exercises

1. Implement a binary classifier for the Breast Cancer dataset
2. Build a multi-class classifier for handwritten digit recognition
3. Create a multi-label classifier for movie genre prediction
4. Implement weighted loss to handle class imbalance
5. Calculate and visualize a confusion matrix for your classifier

## Next Steps

- [Architecture Design](architecture-design.md) - Designing neural network architectures
- [Training & Evaluation](training-evaluation.md) - Training classifiers and evaluating performance

---

**Last Updated**: January 2026
