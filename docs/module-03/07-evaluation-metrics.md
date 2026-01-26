# Evaluation Metrics

## Learning Objectives

By the end of this lesson, you will be able to:
- Calculate accuracy, precision, recall, and F1-score
- Create and interpret confusion matrices
- Use torchmetrics for automated metric calculation
- Understand when to use each metric
- Compare models using multiple metrics

---

## Why Accuracy Isn't Enough

Accuracy is the most intuitive metric, but it can be misleading.

### The Imbalanced Class Problem

```
Example: 1000 emails, 950 legitimate (95%), 50 spam (5%)

Model that predicts "legitimate" for everything:
├─ Accuracy: 95% (looks good!)
├─ But: Misses ALL spam (useless!)
└─ Need: Precision and Recall
```

**Key insight:** When classes are imbalanced, accuracy can be misleading. Use additional metrics.

---

## Confusion Matrix

The confusion matrix is the foundation for all classification metrics.

### Binary Confusion Matrix

```
                    Predicted
               Positive    Negative
Actual  Positive    TP          FN
        Negative    FP          TN

TP = True Positive  (correctly predicted positive)
TN = True Negative  (correctly predicted negative)
FP = False Positive (incorrectly predicted positive)
FN = False Negative (incorrectly predicted negative)
```

### Example: Spam Detection

```
                    Predicted Spam    Predicted Not Spam
Actual Spam              40                  10
Actual Not Spam          5                  945

TP = 40 (spam correctly identified)
FN = 10 (spam missed)
FP = 5 (legitimate incorrectly marked as spam)
TN = 945 (legitimate correctly identified)
```

### Creating a Confusion Matrix

```python
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Get predictions
model.eval()
with torch.inference_mode():
    test_logits = model(X_test)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs > 0.5).long()

# Calculate confusion matrix
cm = confusion_matrix(y_test.numpy(), test_preds.numpy())
print("Confusion Matrix:")
print(cm)
print(f"\nTP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")
```

---

## Key Metrics

### 1. Accuracy

**What:** Percentage of correct predictions

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use:** Balanced datasets (similar class sizes)

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
print(f"Accuracy: {accuracy:.4f}")
```

---

### 2. Precision

**What:** Of all predicted positives, how many are actually positive?

**Formula:**
```
Precision = TP / (TP + FP)
```

**When to use:** When false positives are costly
- Spam detection (don't want to mark legitimate email as spam)
- Medical diagnosis (don't want to falsely diagnose healthy patients)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test.numpy(), test_preds.numpy())
print(f"Precision: {precision:.4f}")
```

---

### 3. Recall (Sensitivity)

**What:** Of all actual positives, how many did we predict correctly?

**Formula:**
```
Recall = TP / (TP + FN)
```

**When to use:** When false negatives are costly
- Disease detection (don't want to miss sick patients)
- Fraud detection (don't want to miss fraudulent transactions)

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test.numpy(), test_preds.numpy())
print(f"Recall: {recall:.4f}")
```

---

### 4. F1-Score

**What:** Harmonic mean of precision and recall

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**When to use:** When you need a balance between precision and recall
- Imbalanced datasets
- When you need a single metric to compare models

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test.numpy(), test_preds.numpy())
print(f"F1-Score: {f1:.4f}")
```

---

## Metric Comparison

### When to Use Each Metric

| Metric | Best For | Example |
|--------|----------|---------|
| **Accuracy** | Balanced datasets | Image classification (equal classes) |
| **Precision** | Minimize false positives | Spam filters, recommendation systems |
| **Recall** | Minimize false negatives | Medical diagnosis, fraud detection |
| **F1-Score** | Imbalanced datasets | Rare event detection |

### Precision vs Recall Trade-off

```
High Precision, Low Recall:
├─ Very confident when predicting positive
├─ Few false positives
└─ But misses many actual positives

Low Precision, High Recall:
├─ Catches most actual positives
├─ Many false positives
└─ Better to be safe than sorry
```

**Example:**
- **Cancer screening:** Optimize for recall (don't miss cancer)
- **Spam filter:** Optimize for precision (don't delete legitimate emails)

---

## Using torchmetrics

`torchmetrics` provides PyTorch-native metric calculation.

### Installation

```bash
pip install torchmetrics
```

### Binary Classification Metrics

```python
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

# Setup metrics
accuracy = Accuracy(task='binary')
precision = Precision(task='binary')
recall = Recall(task='binary')
f1 = F1Score(task='binary')
confmat = ConfusionMatrix(task='binary')

# Calculate metrics
model.eval()
with torch.inference_mode():
    test_logits = model(X_test)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs > 0.5).long()

    # Update metrics with predictions
    acc = accuracy(test_preds, y_test)
    prec = precision(test_preds, y_test)
    rec = recall(test_preds, y_test)
    f1_score = f1(test_preds, y_test)
    cm = confmat(test_preds, y_test)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Confusion Matrix:\n{cm}")
```

### Multi-class Metrics

```python
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

# For multi-class classification
num_classes = 4

accuracy = Accuracy(task='multiclass', num_classes=num_classes)
precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
confmat = ConfusionMatrix(task='multiclass', num_classes=num_classes)

# Calculate metrics (assuming model outputs logits)
model.eval()
with torch.inference_mode():
    test_logits = model(X_test)
    test_probs = torch.softmax(test_logits, dim=1)
    test_preds = torch.argmax(test_probs, dim=1)

    acc = accuracy(test_preds, y_test)
    prec = precision(test_preds, y_test)
    rec = recall(test_preds, y_test)
    f1_score = f1(test_preds, y_test)
    cm = confmat(test_preds, y_test)

print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1-Score (macro): {f1_score:.4f}")
print(f"Confusion Matrix:\n{cm}")
```

### Averaging Methods for Multi-class

| Average | Description | When to Use |
|---------|-------------|-------------|
| **micro** | Average over all instances | When class size matters |
| **macro** | Average over classes (equal weight) | When all classes equally important |
| **weighted** | Weighted average by class size | When you want to account for imbalance |

```python
# Macro averaging (each class equally important)
precision_macro = Precision(task='multiclass', num_classes=4, average='macro')

# Weighted averaging (account for class imbalance)
precision_weighted = Precision(task='multiclass', num_classes=4, average='weighted')

# Micro averaging (global counts)
precision_micro = Precision(task='multiclass', num_classes=4, average='micro')
```

---

## Visualizing Confusion Matrix

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    """Plot confusion matrix with annotations"""

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Configure axes
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    return ax

# Usage
model.eval()
with torch.inference_mode():
    test_logits = model(X_test)
    test_preds = (torch.sigmoid(test_logits) > 0.5).long()

cm = confusion_matrix(y_test.numpy(), test_preds.numpy())
plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])
plt.show()
```

---

## Complete Evaluation Example

```python
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

def evaluate_classifier(model, X_test, y_test, class_names=None):
    """Comprehensive evaluation of a classifier"""

    # Set model to evaluation mode
    model.eval()

    # Make predictions
    with torch.inference_mode():
        test_logits = model(X_test)

        # Handle binary vs multi-class
        if test_logits.shape[1] == 1:
            # Binary classification
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs > 0.5).long()
        else:
            # Multi-class classification
            test_probs = torch.softmax(test_logits, dim=1)
            test_preds = torch.argmax(test_probs, dim=1)

    # Convert to numpy
    y_true = y_test.numpy()
    y_pred = test_preds.numpy()

    # Calculate metrics
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary' if len(set(y_true)) == 2 else 'macro')
    recall = recall_score(y_true, y_pred, average='binary' if len(set(y_true)) == 2 else 'macro')
    f1 = f1_score(y_true, y_pred, average='binary' if len(set(y_true)) == 2 else 'macro')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Detailed classification report
    print("\n" + "=" * 60)
    print("DETAILED REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Usage
results = evaluate_classifier(
    model,
    X_test,
    y_test,
    class_names=['Class 0', 'Class 1']
)
```

---

## Key Takeaways

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced datasets |
| **Precision** | TP/(TP+FP) | Minimize false positives |
| **Recall** | TP/(TP+FN) | Minimize false negatives |
| **F1-Score** | 2×(Prec×Rec)/(Prec+Rec) | Imbalanced datasets |
| **Confusion Matrix** | Visual breakdown | Always use for insight |

| Tool | Pros | Cons |
|------|------|------|
| **sklearn.metrics** | Simple, familiar | Not PyTorch-native |
| **torchmetrics** | PyTorch-native, GPU support | Requires installation |

---

## Discussion Questions

1. **When would precision be more important than recall?** Give a real-world example.

2. **What happens to accuracy if you have 99% negative samples and predict all negatives?** What metrics would reveal the problem?

3. **Why use macro averaging instead of weighted averaging for multi-class metrics?** When would each be appropriate?

---

## Practice Exercises

1. **Calculate metrics manually:**
   - Given TP=50, TN=940, FP=5, FN=5
   - Calculate accuracy, precision, recall, F1

2. **Compare models:**
   - Train two different models
   - Compare using accuracy, precision, recall, F1
   - Which metric is most important for your use case?

3. **Visualize confusion matrix:**
   - Make predictions on test set
   - Create confusion matrix
   - Plot with annotations

---

## Next Steps

- [Practice Exercise](../../module-03/neural-network-classification/05_evaluation_metrics.py) - Hands-on with metrics
- [Complete Workflow](../../module-03/neural-network-classification/06_complete_classification_workflow.py) - End-to-end project

---

**Last Updated:** January 2026
