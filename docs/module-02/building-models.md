# Building Neural Networks

## Learning Objectives

By the end of this lesson, you will be able to:
- Build neural networks using `nn.Module`
- Use common layer types (Linear, Conv2d, etc.)
- Understand activation functions
- Implement forward propagation
- Create complex architectures by combining layers

## Introduction to nn.Module

`nn.Module` is the base class for all neural network modules in PyTorch. It provides:

- **Parameter management**: Automatically tracks learnable parameters
- **GPU support**: Easy movement between CPU and GPU
- **State management**: Handles model state (train/eval)
- **Nesting**: Combine modules to create complex architectures

### Basic Module Structure

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define layers here

    def forward(self, x):
        # Define forward pass here
        return x
```

> **Note**: The `__init__` method defines the layers, while `forward` defines how data flows through them.

## Common Layer Types

### Linear (Fully Connected) Layers

```python
import torch.nn as nn

# Single linear layer: y = xA^T + b
linear = nn.Linear(in_features=10, out_features=5)

# Input: (batch_size, 10)
x = torch.randn(32, 10)

# Output: (batch_size, 5)
output = linear(x)
print("Linear output shape:", output.shape)  # torch.Size([32, 5])
```

### Convolutional Layers

```python
# 2D Convolution for images
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=16,    # 16 output channels
    kernel_size=3,      # 3x3 kernel
    stride=1,           # Stride of 1
    padding=1           # Padding of 1
)

# Input: (batch_size, 3, 28, 28)
x = torch.randn(32, 3, 28, 28)

# Output: (batch_size, 16, 28, 28)
output = conv(x)
print("Conv2d output shape:", output.shape)  # torch.Size([32, 16, 28, 28])
```

### Pooling Layers

```python
# Max pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Input: (batch_size, 16, 28, 28)
x = torch.randn(32, 16, 28, 28)

# Output: (batch_size, 16, 14, 14)
output = maxpool(x)
print("MaxPool output shape:", output.shape)  # torch.Size([32, 16, 14, 14])

# Average pooling
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# Adaptive pooling (output size is fixed)
adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
```

### Normalization Layers

```python
# Batch Normalization
batch_norm = nn.BatchNorm2d(num_features=16)

# Layer Normalization
layer_norm = nn.LayerNorm(normalized_shape=16)

# Dropout (for regularization)
dropout = nn.Dropout(p=0.5)  # 50% dropout rate
```

### Recurrent Layers

```python
# RNN
rnn = nn.RNN(
    input_size=10,
    hidden_size=32,
    num_layers=2,
    batch_first=True
)

# LSTM
lstm = nn.LSTM(
    input_size=10,
    hidden_size=32,
    num_layers=2,
    batch_first=True
)

# GRU
gru = nn.GRU(
    input_size=10,
    hidden_size=32,
    num_layers=2,
    batch_first=True
)

# Input: (batch_size, sequence_length, input_size)
x = torch.randn(32, 10, 10)

# Output: (batch_size, sequence_length, hidden_size)
output, hidden = lstm(x)
print("LSTM output shape:", output.shape)  # torch.Size([32, 10, 32])
```

## Activation Functions

### Common Activation Functions

| Activation | Formula | Use Case |
|------------|---------|----------|
| **ReLU** | `max(0, x)` | Hidden layers (default choice) |
| **Sigmoid** | `1 / (1 + e^(-x))` | Binary classification output |
| **Tanh** | `(e^x - e^(-x)) / (e^x + e^(-x))` | Hidden layers (rare) |
| **Softmax** | `exp(x_i) / sum(exp(x))` | Multi-class classification output |
| **LeakyReLU** | `max(0.01x, x)` | Hidden layers (addresses dying ReLU) |
| **GELU** | `x * Phi(x)` | Transformer models |

### Using Activation Functions

```python
import torch.nn as nn
import torch.nn.functional as F

# Method 1: As modules (in __init__)
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)

# Method 2: As functional calls (in forward)
def forward(self, x):
    x = F.relu(x)
    x = F.sigmoid(x)
    x = F.softmax(x, dim=1)
    return x
```

### Activation Function Examples

```python
x = torch.randn(32, 10)

# ReLU
relu = nn.ReLU()
output = relu(x)  # All negative values become 0

# LeakyReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
output = leaky_relu(x)  # Small negative slope

# Softmax (for multi-class classification)
softmax = nn.Softmax(dim=1)
output = softmax(x)  # Each row sums to 1

# LogSoftmax (more numerically stable)
log_softmax = nn.LogSoftmax(dim=1)
output = log_softmax(x)  # Log probabilities
```

## Building Complete Networks

### Example 1: Simple Feedforward Network

```python
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Forward pass
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Usage
model = FeedForwardNet(input_size=784, hidden_size=128, num_classes=10)
x = torch.randn(32, 784)  # Batch of 32 images (28x28 flattened)
output = model(x)
print("Output shape:", output.shape)  # torch.Size([32, 10])
```

### Example 2: Convolutional Neural Network

```python
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        # Input: 3x224x224 -> After 3 poolings: 128x28x28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 112x112

        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 56x56

        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # 28x28

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Usage
model = ConvNet(num_classes=10)
x = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
output = model(x)
print("Output shape:", output.shape)  # torch.Size([4, 10])
```

### Example 3: Sequential Model

```python
# Using nn.Sequential for simple architectures
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)

# Usage
x = torch.randn(32, 784)
output = model(x)
print("Output shape:", output.shape)  # torch.Size([32, 10])
```

### Example 4: Model with Multiple Paths

```python
class MultiPathNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiPathNet, self).__init__()

        # Shared layer
        self.shared = nn.Linear(input_size, 64)

        # Path 1: Deep but narrow
        self.path1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Path 2: Wide
        self.path2 = nn.Linear(64, 16)

        # Combine paths
        self.combined = nn.Linear(32, num_classes)

    def forward(self, x):
        # Shared features
        x = self.shared(x)
        x = F.relu(x)

        # Two paths
        out1 = self.path1(x)
        out2 = self.path2(x)

        # Concatenate paths
        combined = torch.cat([out1, out2], dim=1)

        # Final layer
        out = self.combined(combined)
        return out

# Usage
model = MultiPathNet(input_size=100, num_classes=10)
x = torch.randn(32, 100)
output = model(x)
print("Output shape:", output.shape)  # torch.Size([32, 10])
```

## Model Inspection

### Checking Model Structure

```python
model = FeedForwardNet(784, 128, 10)

# Print model architecture
print(model)

# Print named parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### Model Summary

```python
from torchsummary import summary

model = ConvNet(num_classes=10)
summary(model, input_size=(3, 224, 224))
```

## Model Modes

### Train vs Eval Mode

```python
model = ConvNet(num_classes=10)

# Set to training mode
model.train()
# Enables dropout, batch normalization updates, etc.

# Set to evaluation mode
model.eval()
# Disables dropout, uses running statistics for batch norm

# Example: Training loop
model.train()
for batch in train_loader:
    # Training code
    pass

# Example: Validation loop
model.eval()
with torch.no_grad():  # Disable gradient computation
    for batch in val_loader:
        # Validation code
        pass
```

## Moving Models Between Devices

```python
model = ConvNet(num_classes=10)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Move input to device
x = torch.randn(4, 3, 224, 224).to(device)

# Forward pass
output = model(x)

# Common pattern for device-agnostic code
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()
model = model.to(device)
```

## Best Practices

| Practice | Description |
|----------|-------------|
| **Use `nn.Sequential`** for simple, linear stacks of layers |
| **Custom `nn.Module`** for complex architectures with multiple paths |
| **Batch Normalization** after convolutional/linear layers, before activation |
| **Dropout** after activation functions |
| **Activation functions** typically after normalization |
| **Model mode** always set correctly (train/eval) |

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **nn.Module** is the base class for all models | Implement `__init__` and `forward` |
| **Layers** are defined in `__init__` | Linear, Conv2d, MaxPool2d, etc. |
| **Forward pass** defines data flow | Use layers defined in `__init__` |
| **Activation functions** add non-linearity | ReLU, Sigmoid, Softmax, etc. |
| **Model modes** control behavior | `train()` for training, `eval()` for inference |

## Practice Exercises

1. Create a feedforward network with 3 hidden layers for binary classification
2. Build a CNN with at least 3 convolutional layers for image classification
3. Implement a model with skip connections (residual connections)
4. Create a model that processes both image and text inputs
5. Count the number of parameters in a model and identify which layers have the most

## Next Steps

- [Data Preparation](data-preparation.md) - Loading and preprocessing data
- [Training Loop Fundamentals](training-loop.md) - Implementing the training process

---

**Last Updated**: January 2026
