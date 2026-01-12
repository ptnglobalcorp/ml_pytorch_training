# Model Persistence

## Learning Objectives

By the end of this lesson, you will be able to:
- Save and load PyTorch models
- Save and load optimizer states
- Use checkpoints for training resumption
- Export models for production deployment
- Handle device compatibility when loading models

## Introduction to Model Persistence

PyTorch provides two main approaches for saving models:
1. **Saving only the model parameters** (state_dict) - Recommended
2. **Saving the entire model** - Less flexible but simpler

> **Best Practice**: Always prefer saving `state_dict` as it's more flexible and portable.

## Saving and Loading State Dict

### What is a state_dict?

A `state_dict` is a Python dictionary that maps each layer to its parameter tensors:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)

# View state dict
print(model.state_dict())
# OrderedDict([
#     ('weight', tensor(...)),
#     ('bias', tensor(...))
# ])

# Access specific parameter
print(model.state_dict()['weight'].shape)  # torch.Size([5, 10])
```

### Saving a Model

```python
# Method 1: Save state dict (recommended)
torch.save(model.state_dict(), 'model.pth')

# Method 2: Save entire model (less flexible)
torch.save(model, 'entire_model.pth')

# Method 3: Save with additional information
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

### Loading a Model

```python
# Method 1: Load state dict (recommended)
model = ModelClass(*args, **kwargs)  # Initialize model
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode

# Method 2: Load entire model
model = torch.load('entire_model.pth')
model.eval()

# Method 3: Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()
```

## Saving During Training

### Creating Checkpoints

```python
def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save a training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved: {filename}')


# Usage during training
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, epoch, train_loss, f'checkpoint_epoch_{epoch}.pth')

    # Save best model
    if train_loss < best_loss:
        best_loss = train_loss
        save_checkpoint(model, optimizer, epoch, train_loss, 'best_model.pth')
```

### Resuming Training

```python
def load_checkpoint(model, optimizer, filename):
    """Load a training checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


# Usage
model = ModelClass(input_size, hidden_size, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load checkpoint if exists
checkpoint_path = 'checkpoint_epoch_10.pth'
if os.path.exists(checkpoint_path):
    start_epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)
    print(f'Resuming from epoch {start_epoch}')
else:
    start_epoch = 0
    print('Starting training from scratch')

# Continue training
for epoch in range(start_epoch, num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    # ...
```

## Device Handling

### Saving for Different Devices

```python
# Save on CPU (most compatible)
torch.save(model.state_dict(), 'model_cpu.pth', _use_new_zipfile_serialization=True)

# Save on GPU
model.to('cuda')
torch.save(model.state_dict(), 'model_gpu.pth')
```

### Loading on Different Devices

```python
# Load on CPU (works regardless of where it was saved)
device = torch.device('cpu')
model = ModelClass(*args, **kwargs)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)

# Load on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelClass(*args, **kwargs)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)

# Load GPU model on CPU (force)
model.load_state_dict(torch.load('model_gpu.pth', map_location='cpu'))

# Load CPU model on GPU (if available)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('model_cpu.pth', map_location='cuda'))
```

### Device-Agnostic Loading

```python
def load_model(model_class, model_path, *args, **kwargs):
    """Load model on the best available device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device

# Usage
model, device = load_model(MyModel, 'model.pth', input_size=10, num_classes=5)
```

## Model Export

### TorchScript Export

TorchScript allows you to export models for production deployment:

```python
# Method 1: Tracing (captures the actual computation)
model = MyModel()
model.eval()

# Example input
example_input = torch.randn(1, 3, 224, 224)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
traced_model.save('traced_model.pt')

# Load and use
loaded_model = torch.jit.load('traced_model.pt')
output = loaded_model(example_input)

# Method 2: Scripting (captures the entire model logic)
scripted_model = torch.jit.script(model)
scripted_model.save('scripted_model.pt')
```

### ONNX Export

Export to ONNX format for deployment in other frameworks:

```python
# Export to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,                          # Model to export
    dummy_input,                    # Example input
    "model.onnx",                   # Output filename
    export_params=True,             # Store trained parameters
    opset_version=14,               # ONNX version
    do_constant_folding=True,       # Optimize constants
    input_names=['input'],          # Input names
    output_names=['output'],        # Output names
    dynamic_axes={                  # Dynamic axes for variable length
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## Complete Example

### Training with Checkpointing

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_val_acc = 0.0

        # Create checkpoint directory
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, targets in self.train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')

    def load_checkpoint(self, filename):
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        print(f'Checkpoint loaded: {filepath}, resuming from epoch {start_epoch}')
        return start_epoch

    def train(self, num_epochs, resume_from=None):
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 'best_model.pth')

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')

            print('-' * 60)

        return self.model

# Usage
model = ConvNet(num_classes=10)
trainer = Trainer(model, train_loader, val_loader, device)
trained_model = trainer.train(num_epochs=20, resume_from='checkpoint_epoch_10.pth')
```

## Model Versioning

### Organizing Model Artifacts

```
models/
├── experiment_1/
│   ├── best_model.pth
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_5.pth
│   │   ├── checkpoint_epoch_10.pth
│   │   └── checkpoint_epoch_15.pth
│   └── metadata.json
├── experiment_2/
│   └── ...
└── production/
    └── model_v1.0.pt
```

### Saving Metadata

```python
import json
from datetime import datetime

def save_model_with_metadata(model, optimizer, metrics, model_dir, model_name):
    """Save model with metadata"""
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)

    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'model architecture': str(model),
        'parameters': sum(p.numel() for p in model.parameters()),
    }

    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Model and metadata saved to {model_dir}')

# Usage
save_model_with_metadata(
    model=model,
    optimizer=optimizer,
    metrics={'val_acc': 95.2, 'val_loss': 0.15},
    model_dir='models/experiment_1',
    model_name='best_model'
)
```

## Best Practices

| Practice | Description |
|----------|-------------|
| **Save state_dict** | More flexible than saving entire model |
| **Save checkpoints** | Save model + optimizer + epoch for resuming |
| **Track best model** | Save separate copy of best performing model |
| **Device-agnostic loading** | Use `map_location` when loading |
| **Version control** | Save model metadata and configuration |
| **Regular backups** | Save checkpoints at regular intervals |
| **Validate after loading** | Always verify loaded model works correctly |

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **state_dict** contains all learnable parameters | Use `model.state_dict()` and `model.load_state_dict()` |
| **Checkpoints** save training state | Include model, optimizer, epoch, and metrics |
| **Device handling** requires care | Use `map_location` for cross-device loading |
| **Model export** enables deployment | Use TorchScript or ONNX for production |
| **Metadata** aids reproducibility | Save configuration and metrics with models |

## Practice Exercises

1. Create a function that saves a model with metadata
2. Implement a training loop with periodic checkpointing every N epochs
3. Write code to resume training from a checkpoint
4. Export a model to ONNX format
5. Create a model versioning system that tracks training metadata

## Next Steps

- [Data Preparation](data-preparation.md) - Loading and preprocessing data
- [Building Neural Networks](building-models.md) - Creating models with nn.Module
- [Training Loop Fundamentals](training-loop.md) - Implementing the training process
- [Module 3: Neural Network Classification](../module-03/README.md) - Building classifiers

---

**Last Updated**: January 2026
