"""
Exercise 2: Building Neural Networks
PyTorch Workflow Fundamentals - Module 2

This exercise covers:
- Creating models with nn.Module
- Using common layer types
- Implementing forward propagation
- Understanding activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# Part 1: Basic Neural Network
# ============================================

print("=" * 60)
print("Part 1: Basic Neural Network")
print("=" * 60)


class SimpleNet(nn.Module):
    """A simple feedforward neural network"""

    def __init__(self, input_size, hidden_size, num_classes):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_classes: Number of output classes
        """
        super(SimpleNet, self).__init__()

        # TODO: Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # TODO: Implement forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# TODO: Create an instance of SimpleNet
model = SimpleNet(input_size=20, hidden_size=64, num_classes=5)

# TODO: Print model architecture
print(model)

# TODO: Create a random input and pass it through the model
input_tensor = torch.randn(10, 20)  # Batch size 10
output = model(input_tensor)
print(f"\nInput shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")

# TODO: Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")


# ============================================
# Part 2: Neural Network with Batch Normalization and Dropout
# ============================================

print("\n" + "=" * 60)
print("Part 2: Neural Network with Batch Norm and Dropout")
print("=" * 60)


class RegularizedNet(nn.Module):
    """Network with batch normalization and dropout"""

    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(RegularizedNet, self).__init__()

        # TODO: Build layers dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            # ReLU activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        # TODO: Create sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.network(x)


# TODO: Create an instance of RegularizedNet
model = RegularizedNet(
    input_size=20,
    hidden_sizes=[64, 32],
    num_classes=5,
    dropout_rate=0.3
)

print(model)

# TODO: Test the model
input_tensor = torch.randn(10, 20)
output = model(input_tensor)
print(f"\nInput shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")


# ============================================
# Part 3: Convolutional Neural Network
# ============================================

print("\n" + "=" * 60)
print("Part 3: Convolutional Neural Network")
print("=" * 60)


class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # TODO: Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # TODO: Define pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # TODO: Define fully connected layers
        # After 3 poolings: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # TODO: Define activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        """
        # TODO: Implement forward pass
        # Conv block 1
        x = self.pool(self.relu(self.conv1(x)))  # -> 112x112

        # Conv block 2
        x = self.pool(self.relu(self.conv2(x)))  # -> 56x56

        # Conv block 3
        x = self.pool(self.relu(self.conv3(x)))  # -> 28x28

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# TODO: Create an instance of SimpleCNN
model = SimpleCNN(num_classes=10)

print(model)

# TODO: Test the model
input_tensor = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
output = model(input_tensor)
print(f"\nInput shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")


# ============================================
# Part 4: Model with Skip Connection
# ============================================

print("\n" + "=" * 60)
print("Part 4: Model with Skip Connection")
print("=" * 60)


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # TODO: Define main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # TODO: Define skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # TODO: Implement forward pass with skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # TODO: Add skip connection
        out += self.skip(x)

        out = self.relu(out)
        return out


# TODO: Test residual block
block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
input_tensor = torch.randn(1, 64, 56, 56)
output = block(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Create a model for binary classification
print("\nExercise 1: Binary classification model")
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        # TODO: Implement this
        pass

    def forward(self, x):
        # TODO: Implement this
        pass

# Exercise 2: Implement a model with multiple output heads
print("\nExercise 2: Multi-output model")
class MultiOutputModel(nn.Module):
    """Model with multiple output heads for different tasks"""
    def __init__(self, input_size, shared_size, task1_size, task2_size):
        super(MultiOutputModel, self).__init__()
        # TODO: Implement shared layers and task-specific heads
        pass

    def forward(self, x):
        # TODO: Return outputs for both tasks
        pass

# Exercise 3: Create a model that accepts variable-length inputs
print("\nExercise 3: Variable-length input model")
# Use GlobalAveragePooling to handle variable input sizes

# Exercise 4: Implement a Siamese network for similarity learning
print("\nExercise 4: Siamese network")
# Create a network that processes two inputs and compares them

# Exercise 5: Build a model that uses different activation functions
print("\nExercise 5: Mixed activation functions")
# Create a model that uses ReLU, LeakyReLU, and GELU in different layers


print("\n" + "=" * 60)
print("Exercise 2 Complete!")
print("=" * 60)
