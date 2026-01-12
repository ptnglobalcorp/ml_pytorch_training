# Training Loop Fundamentals

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the complete training workflow
- Implement loss functions for different tasks
- Use optimizers for parameter updates
- Write effective training and validation loops
- Track metrics and monitor training progress

## The Training Workflow

The complete training process in PyTorch follows these steps:

```
1. Prepare data (Dataset + DataLoader)
2. Define model (nn.Module)
3. Define loss function
4. Define optimizer
5. Training loop:
   - Forward pass
   - Compute loss
   - Backward pass (compute gradients)
   - Update parameters
   - Zero gradients
6. Validation loop (evaluate on test data)
```

## Loss Functions

### Regression Losses

```python
import torch.nn as nn

# Mean Squared Error (MSE)
mse_loss = nn.MSELoss()
predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)
loss = mse_loss(predictions, targets)

# Mean Absolute Error (MAE)
mae_loss = nn.L1Loss()
loss = mae_loss(predictions, targets)

# Smooth L1 Loss (Huber loss)
smooth_loss = nn.SmoothL1Loss()
loss = smooth_loss(predictions, targets)
```

### Classification Losses

```python
# Binary Cross Entropy (for binary classification)
# Note: Input should be probabilities (after sigmoid)
bce_loss = nn.BCELoss()
predictions = torch.sigmoid(torch.randn(32, 1))
targets = torch.randint(0, 2, (32, 1)).float()
loss = bce_loss(predictions, targets)

# Binary Cross Entropy with Logits (more stable)
# Note: Input should be raw logits (before sigmoid)
bce_logits_loss = nn.BCEWithLogitsLoss()
logits = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()
loss = bce_logits_loss(logits, targets)

# Cross Entropy Loss (for multi-class classification)
# Note: Input should be raw logits (before softmax)
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)  # 10 classes
targets = torch.randint(0, 10, (32,))
loss = ce_loss(logits, targets)

# Negative Log Likelihood (for multi-class with log_softmax)
nll_loss = nn.NLLLoss()
log_probs = torch.log_softmax(torch.randn(32, 10), dim=1)
targets = torch.randint(0, 10, (32,))
loss = nll_loss(log_probs, targets)
```

### Choosing the Right Loss Function

| Task | Loss Function | Input Format |
|------|---------------|--------------|
| **Regression** | `MSELoss` | Raw predictions |
| **Binary Classification** | `BCEWithLogitsLoss` | Logits |
| **Multi-class Classification** | `CrossEntropyLoss` | Logits |
| **Multi-label Classification** | `BCEWithLogitsLoss` | Logits per class |

## Optimizers

### Common Optimizers

```python
import torch.optim as optim

model = nn.Linear(10, 1)

# Stochastic Gradient Descent
optimizer_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

# Adam (most commonly used)
optimizer_adam = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)
)

# AdamW (Adam with weight decay)
optimizer_adamw = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

# RMSprop
optimizer_rmsprop = optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99
)
```

### Optimizer Comparison

| Optimizer | Description | When to Use |
|-----------|-------------|-------------|
| **SGD** | Simple, works well with momentum | Baseline, large-batch training |
| **Adam** | Adaptive learning rates | Most tasks, default choice |
| **AdamW** | Adam with decoupled weight decay | Transformers, NLP |
| **RMSprop** | Similar to Adam, simpler | RNNs, some vision tasks |

### Learning Rate Scheduling

```python
# Step LR: decay by gamma every step_size epochs
scheduler_step = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,  # Decay every 10 epochs
    gamma=0.1      # Multiply LR by 0.1
)

# Reduce on Plateau: reduce when metric stops improving
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Minimize validation loss
    factor=0.1,      # Reduce by factor of 10
    patience=5       # Wait 5 epochs before reducing
)

# Cosine Annealing
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50  # Epochs to complete one cycle
)
```

## The Training Loop

### Basic Training Loop Structure

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardNet(input_size=784, hidden_size=128, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Training loop
num_epochs = 10
model.train()  # Set model to training mode

for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device
        data = data.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Print epoch statistics
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
```

### Complete Training Loop with Validation

```python
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """Complete training loop"""
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy: {val_acc:.2f}%')

        print('-' * 60)

    return train_losses, train_accs, val_losses, val_accs
```

## Training Best Practices

### Gradient Clipping

Prevent exploding gradients in RNNs or deep networks:

```python
# Clip gradients by norm during training
max_norm = 1.0
for data, targets in train_loader:
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    optimizer.step()
```

### Early Stopping

Stop training when validation loss stops improving:

```python
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

# Usage
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

### Mixed Precision Training

Use mixed precision for faster training on modern GPUs:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, targets in train_loader:
    data, targets = data.to(device), targets.to(device)

    optimizer.zero_grad()

    # Automatic mixed precision
    with autocast():
        outputs = model(data)
        loss = criterion(outputs, targets)

    # Scale gradients and update
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Monitoring Training

### Tracking Metrics

```python
# Simple tracking
train_losses = []
val_losses = []
learning_rates = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])
```

### Using TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/experiment_1')

# Log metrics during training
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # Log histograms
    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param, epoch)
        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

writer.close()
```

## Practical Examples

### Example 1: Training a Classifier

```python
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Training
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_checkpoint.pth')

    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```

## Key Takeaways

| Component | Description |
|-----------|-------------|
| **Loss Function** measures prediction error | CrossEntropy for classification, MSE for regression |
| **Optimizer** updates parameters | Adam, SGD, AdamW are most common |
| **Training Loop** iterates over data | Forward, loss, backward, update |
| **Validation Loop** evaluates performance | No gradient computation, different metrics |
| **Scheduler** adjusts learning rate | StepLR, ReduceLROnPlateau, CosineAnnealing |

## Practice Exercises

1. Implement a training loop for a regression task with MSE loss
2. Add learning rate scheduling to an existing training loop
3. Implement early stopping based on validation loss
4. Create a training loop that logs metrics to TensorBoard
5. Implement gradient clipping for a recurrent neural network

## Next Steps

- [Model Persistence](model-persistence.md) - Saving and loading trained models
- [Module 3: Neural Network Classification](../module-03/README.md) - Building classifiers

---

**Last Updated**: January 2026
