# Architecture Design

## Learning Objectives

By the end of this lesson, you will be able to:
- Design neural network architectures for classification
- Understand common architectural patterns
- Implement skip connections (residual networks)
- Design convolutional neural networks (CNNs)
- Apply design principles for better model performance

## Architecture Design Principles

When designing neural network architectures, consider these principles:

| Principle | Description | Application |
|-----------|-------------|-------------|
| **Depth** | More layers can learn more complex features | Add layers for complex patterns |
| **Width** | More neurons per layer increases capacity | Use wider layers for rich features |
| **Skip Connections** | Direct connections between non-adjacent layers | Enables deeper networks, better gradient flow |
| **Normalization** | Batch/layer normalization stabilizes training | Add after linear/conv layers |
| **Dropout** | Regularization to prevent overfitting | Add after activation functions |

## Simple Feedforward Architecture

### Design Pattern

```
Input Layer
    ↓
Hidden Layer 1 (Linear + ReLU + BatchNorm + Dropout)
    ↓
Hidden Layer 2 (Linear + ReLU + BatchNorm + Dropout)
    ↓
Output Layer (Linear + Softmax/Sigmoid)
```

### Implementation

```python
import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(FeedForwardNetwork, self).__init__()

        # Build hidden layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Usage
model = FeedForwardNetwork(
    input_size=784,
    hidden_sizes=[256, 128, 64],
    num_classes=10,
    dropout_rate=0.3
)
print(model)
```

## Convolutional Neural Network Architecture

### Design Pattern

```
Input (Images)
    ↓
Conv Block 1 (Conv + BatchNorm + ReLU + MaxPool)
    ↓
Conv Block 2 (Conv + BatchNorm + ReLU + MaxPool)
    ↓
Conv Block 3 (Conv + BatchNorm + ReLU + MaxPool)
    ↓
Flatten
    ↓
Fully Connected Layers (Linear + ReLU + Dropout)
    ↓
Output Layer
```

### Implementation

```python
class ConvBlock(nn.Module):
    """Reusable convolutional block"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 32),    # Output: 32 x 112 x 112
            ConvBlock(32, 64),   # Output: 64 x 56 x 56
            ConvBlock(64, 128),  # Output: 128 x 28 x 28
            ConvBlock(128, 256), # Output: 256 x 14 x 14
        )

        # Calculate flattened size
        self.flattened_size = 256 * 14 * 14

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Usage
model = CNN(num_classes=10)
print(model)

# Test with dummy input
x = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
output = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([4, 10])
```

## Residual Networks (ResNet)

Skip connections allow gradients to flow through the network more easily, enabling deeper networks.

### Residual Block

```python
class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (if dimensions change)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        out += self.skip(x)
        out = self.relu(out)

        return out
```

### ResNet Architecture

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        Args:
            block: ResidualBlock type
            layers: Number of blocks in each layer
            num_classes: Number of output classes
        """
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ResNet-18
def resnet18(num_classes=10):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

# Usage
model = resnet18(num_classes=10)
print(model)
```

## Architectural Patterns

### Pattern 1: Bottleneck Architecture

Reduce computation by using bottleneck layers (1x1 convolutions):

```python
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(BottleneckBlock, self).__init__()

        reduced_channels = out_channels // reduction

        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(reduced_channels)

        self.conv2 = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(reduced_channels)

        self.conv3 = nn.Conv2d(reduced_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
```

### Pattern 2: Inception-like Module

Parallel pathways with different kernel sizes:

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        """
        Args:
            out_channels_list: [1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj]
        """
        super(InceptionModule, self).__init__()

        # 1x1 conv branch
        self.branch1 = nn.Conv2d(in_channels, out_channels_list[0], kernel_size=1)

        # 1x1 -> 3x3 branch
        self.branch2_conv1 = nn.Conv2d(in_channels, out_channels_list[1], kernel_size=1)
        self.branch2_conv2 = nn.Conv2d(out_channels_list[1], out_channels_list[2], kernel_size=3, padding=1)

        # 1x1 -> 5x5 branch
        self.branch3_conv1 = nn.Conv2d(in_channels, out_channels_list[3], kernel_size=1)
        self.branch3_conv2 = nn.Conv2d(out_channels_list[3], out_channels_list[4], kernel_size=5, padding=2)

        # 3x3 pool -> 1x1 conv branch
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = nn.Conv2d(in_channels, out_channels_list[5], kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2_conv2(self.branch2_conv1(x))
        branch3 = self.branch3_conv2(self.branch3_conv1(x))
        branch4 = self.branch4_conv(self.branch4_pool(x))

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)
```

### Pattern 3: Dense Connectivity

Each layer connects to all subsequent layers:

```python
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.bn1(x)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))
        return torch.cat([x, new_features], dim=1)
```

## Architecture Selection Guide

| Task | Recommended Architecture | Reason |
|------|-------------------------|--------|
| **Simple tabular data** | Feedforward Network | No spatial structure |
| **Small images** | Simple CNN (3-4 conv layers) | Sufficient capacity |
| **Medium images** | ResNet-18/34 | Good balance of depth and efficiency |
| **Large images** | ResNet-50+ or EfficientNet | Deep architectures needed |
| **Real-time inference** | MobileNet or ShuffleNet | Optimized for speed |
| **Limited data** | Transfer learning | Pre-trained features help |

## Design Tips

### Tip 1: Start Simple

```python
# Start with a simple architecture
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Only add complexity if needed
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)
```

### Tip 2: Monitor Model Capacity

```python
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = CNN(num_classes=10)
num_params = count_parameters(model)
print(f"Number of parameters: {num_params:,}")

# Rule of thumb: You need ~10x more training samples than parameters
```

### Tip 3: Use Appropriate Initialization

```python
def initialize_weights(m):
    """Initialize weights using He initialization for ReLU networks"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply to model
model = CNN(num_classes=10)
model.apply(initialize_weights)
```

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Depth** enables learning complex features | Add layers for capacity |
| **Skip connections** enable deeper networks | ResNet-style connections improve gradient flow |
| **Batch normalization** stabilizes training | Add after linear/conv layers |
| **Dropout** prevents overfitting | Add after activation functions |
| **Start simple** and increase complexity | Don't over-engineer initially |
| **Monitor capacity** relative to data | Avoid overfitting with too many parameters |

## Practice Exercises

1. Design a CNN with at least 5 convolutional layers for image classification
2. Implement a residual block with a bottleneck design
3. Create a custom architecture that combines multiple pathways
4. Design a network optimized for mobile inference (few parameters)
5. Compare architectures of different depths on the same dataset

## Next Steps

- [Classification Basics](classification-basics.md) - Understanding classification tasks
- [Training & Evaluation](training-evaluation.md) - Training and evaluating classifiers

---

**Last Updated**: January 2026
