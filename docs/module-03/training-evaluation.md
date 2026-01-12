# Training & Evaluation

## Learning Objectives

By the end of this lesson, you will be able to:
- Implement a complete training pipeline for classification
- Use proper evaluation metrics for classifiers
- Handle common training issues
- Apply regularization techniques
- Perform hyperparameter tuning

## Complete Training Pipeline

### Pipeline Structure

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class ClassificationTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, num_classes=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for data, targets in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(data)

            # Handle different output formats
            if outputs.dim() > 1 and outputs.size(1) == 1:
                # Binary classification
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets.float())
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            else:
                # Multi-class classification
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * data.size(0)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for data, targets in pbar:
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(data)

                # Handle different output formats
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    # Binary classification
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, targets.float())
                    predicted = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    # Multi-class classification
                    loss = criterion(outputs, targets)
                    _, predicted = outputs.max(1)

                # Statistics
                running_loss += loss.item() * data.size(0)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc, all_preds, all_labels

    def test(self, criterion):
        """Test the model and return detailed metrics"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc='Testing'):
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(data)

                # Handle different output formats
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    # Binary classification
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, targets.float())
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).long()
                else:
                    # Multi-class classification
                    loss = criterion(outputs, targets)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                running_loss += loss.item() * data.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)

        if self.num_classes == 2:
            # Binary classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary'
            )
        else:
            # Multi-class metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro'
            )

        confusion_mat = confusion_matrix(all_labels, all_preds)

        results = {
            'test_loss': running_loss / len(all_labels),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_mat,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

        return results

    def train(self, num_epochs, criterion, optimizer, scheduler=None, early_stopping=None):
        """Complete training loop"""
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 60)

            # Training
            train_loss, train_acc = self.train_epoch(criterion, optimizer)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation
            val_loss, val_acc, _, _ = self.validate(criterion)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Print statistics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, 'best_model.pth')
                print(f'✓ New best model saved with accuracy: {val_acc:.2f}%')

            # Early stopping
            if early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print('Early stopping triggered')
                    break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return self.history

    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, cm, class_names=None):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]

        ax.set(
            xlabel='Predicted label',
            ylabel='True label',
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
        plt.show()
```

### Using the Training Pipeline

```python
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(num_classes=10)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3, verbose=True
)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=5, min_delta=0.001)

# Create trainer
trainer = ClassificationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    num_classes=10
)

# Train
history = trainer.train(
    num_epochs=20,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    early_stopping=early_stopping
)

# Plot training history
trainer.plot_history()

# Test
results = trainer.test(criterion)
print(f"\nTest Results:")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1 Score: {results['f1_score']:.4f}")

# Plot confusion matrix
trainer.plot_confusion_matrix(results['confusion_matrix'])
```

## Regularization Techniques

### 1. L1/L2 Regularization (Weight Decay)

```python
# Add weight decay to optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

### 2. Dropout

```python
class RegularizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # 50% dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 30% dropout
            nn.Linear(128, 10)
        )
```

### 3. Data Augmentation (for Images)

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 4. Label Smoothing

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_preds)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        loss = torch.sum(-smooth_target * log_preds, dim=1).mean()
        return loss

# Usage
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

## Handling Common Issues

### Issue 1: Overfitting

**Symptoms**: Training loss decreases, validation loss increases

**Solutions**:
- Add dropout
- Increase weight decay
- Add data augmentation
- Reduce model complexity
- Add more training data

```python
# Increase regularization
model = RegularizedNet()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
```

### Issue 2: Underfitting

**Symptoms**: Both training and validation loss are high

**Solutions**:
- Increase model capacity (more layers/units)
- Reduce regularization (lower weight decay, dropout)
- Train for more epochs
- Try different learning rate

```python
# Increase model capacity
class LargerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 512),  # Larger
            nn.ReLU(),
            nn.Linear(512, 256),  # Additional layer
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
```

### Issue 3: Exploding Gradients

**Symptoms**: Loss becomes NaN, weights explode

**Solutions**:
- Lower learning rate
- Use gradient clipping
- Use batch normalization
- Change initialization

```python
# Gradient clipping
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

### Issue 4: Vanishing Gradients

**Symptoms**: Gradients become very small, training stalls

**Solutions**:
- Use ReLU activation instead of sigmoid/tanh
- Use batch normalization
- Use residual connections
- Use proper initialization (He initialization)

```python
# He initialization for ReLU networks
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
```

## Hyperparameter Tuning

### Manual Tuning

```python
# Try different hyperparameters
configs = [
    {'lr': 0.1, 'weight_decay': 0.0},
    {'lr': 0.01, 'weight_decay': 0.001},
    {'lr': 0.001, 'weight_decay': 0.01},
    {'lr': 0.0001, 'weight_decay': 0.1},
]

for config in configs:
    model = CNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    trainer = ClassificationTrainer(model, train_loader, val_loader, test_loader, device)
    history = trainer.train(num_epochs=10, criterion=criterion, optimizer=optimizer)

    results = trainer.test(criterion)
    print(f"Config: {config}, Test Acc: {results['accuracy']:.4f}")
```

### Grid Search Example

```python
from itertools import product

learning_rates = [0.001, 0.0001]
weight_decays = [0.001, 0.01]
batch_sizes = [32, 64]

best_acc = 0
best_config = None

for lr, wd, bs in product(learning_rates, weight_decays, batch_sizes):
    print(f"Testing: lr={lr}, wd={wd}, bs={bs}")

    # Update dataloader
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    model = CNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    trainer = ClassificationTrainer(model, train_loader, val_loader, test_loader, device)
    trainer.train(num_epochs=5, criterion=criterion, optimizer=optimizer)
    results = trainer.test(criterion)

    if results['accuracy'] > best_acc:
        best_acc = results['accuracy']
        best_config = {'lr': lr, 'wd': wd, 'bs': bs}

print(f"\nBest config: {best_config}")
print(f"Best accuracy: {best_acc:.4f}")
```

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Training Pipeline** encapsulates all steps | Create reusable trainer classes |
| **Regularization** prevents overfitting | Dropout, weight decay, data augmentation |
| **Common Issues** have known solutions | Overfitting, underfitting, gradient problems |
| **Hyperparameter Tuning** finds optimal settings | Systematic search over hyperparameters |
| **Evaluation** uses multiple metrics | Accuracy, precision, recall, F1-score, confusion matrix |

## Practice Exercises

1. Implement a training pipeline for a custom classifier
2. Add data augmentation to an image classifier
3. Implement label smoothing for multi-class classification
4. Perform hyperparameter tuning on a classification task
5. Create a custom callback system for training monitoring

## Next Steps

- [Model Deployment](model-deployment.md) - Deploying trained models
- [Classification Basics](classification-basics.md) - Understanding classification tasks
- [Architecture Design](architecture-design.md) - Designing neural network architectures

---

**Last Updated**: January 2026
