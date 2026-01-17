# Anatomy of a Neural Network

## Learning Objectives

By the end of this lesson, you will be able to:
- Identify key components: Input Layer, Hidden Layers, Output Layer
- Understand the foundational workflow of training a neural network
- Recognize how information flows through networks

## Key Components

A neural network consists of layers of interconnected "neurons" that process and transform data. Let's break down the three essential components:

### Input Layer

The **input layer** is where raw data enters the network. Before data reaches this layer, it must be converted into numerical encoding (tensors).

**Examples of data entering the input layer:**

| Data Type | Raw Input | Tensor Representation |
|-----------|-----------|----------------------|
| Image | 224×224 RGB photo | Tensor of shape `[3, 224, 224]` (channels × height × width) |
| Text | "Hello world" | Sequence of token IDs, shape `[sequence_length]` |
| Tabular | [age, income, score] | Tensor of shape `[3]` (one value per feature) |
| Audio | 5-second audio clip | Tensor of shape `[1, samples]` (channels × audio samples) |

The input layer doesn't perform computation—it simply passes the tensor representation to the first hidden layer.

### Hidden Layers

**Hidden layers** are where the learning happens. These layers of "neurons" (also called hidden units) transform data through a series of linear and non-linear operations.

Each neuron in a hidden layer:
1. Receives inputs from the previous layer (each multiplied by a learned weight)
2. Sums the weighted inputs plus a bias term
3. Applies a non-linear activation function (like ReLU)
4. Passes the result to the next layer

**Why "hidden"?** Because you don't directly observe or control what happens in these layers—they learn patterns automatically from the data.

**What hidden layers learn:**
- **Early hidden layers** (closer to input) detect simple patterns like edges, colors, or basic word combinations
- **Middle hidden layers** combine simple patterns into complex patterns like shapes, objects, or phrases
- **Later hidden layers** (closer to output) combine complex patterns into abstract representations like "dog," "positive sentiment," or "fraudulent transaction"

**Depth matters:**
- **Shallow networks** (1-2 hidden layers) can learn simple relationships
- **Deep networks** (many hidden layers) can learn hierarchical, abstract representations
- The "deep" in deep learning refers to having many hidden layers

### Output Layer

The **output layer** produces the network's final prediction or learned representation.

**Output formats vary by task:**

| Task Type | Output Shape | Interpretation |
|-----------|--------------|----------------|
| Binary Classification | `[1]` or `[2]` | Single probability (0-1) or two class scores |
| Multi-class Classification | `[num_classes]` | Probability distribution over classes |
| Regression | `[num_values]` | Continuous numeric predictions |
| Image Generation | `[C, H, W]` | Generated image tensor |

**Example outputs:**
- **Image classification:** `[0.02, 0.95, 0.03]` → 95% confidence it's class 1 (e.g., "cat")
- **Sentiment analysis:** `[0.85]` → 85% positive sentiment
- **House price prediction:** `[425000]` → Predicted $425,000 value

## Foundational Workflow

How does a neural network learn? It follows an iterative process of making predictions, comparing to actual values, and adjusting internal parameters.

### Step 1: Initialize with Random Weights

When a neural network is first created, all its weights (the parameters connecting neurons between layers) are initialized with random values.

At this stage, the network is essentially making random guesses—it has no knowledge yet. This is like showing an infant a picture and asking them to identify it; they'll guess randomly.

**Why random?** Random initialization breaks symmetry and ensures different neurons learn different features. If all weights started at zero, all neurons would learn identical features.

### Step 2: Show the Model Examples

The network is presented with training data—examples of inputs and their correct outputs.

**A training example consists of:**
- **Input:** The raw data (e.g., an image of a dog)
- **Target:** The correct output (e.g., the label "dog")

The network processes the input through all layers to produce a prediction. This is called the **forward pass**.

**Example forward pass:**
```
Input (image tensor)
    ↓
Hidden Layer 1 (detects edges)
    ↓
Hidden Layer 2 (combines edges into shapes)
    ↓
Hidden Layer 3 (combines shapes into object parts)
    ↓
Output Layer (produces class probabilities)
    ↓
Prediction: [dog: 0.65, cat: 0.25, bird: 0.10]
```

### Step 3: The Model Learns Representations

After making a prediction, the network compares it to the correct answer and calculates a **loss**—a measure of how wrong the prediction was.

**Learning happens through backpropagation:**
1. Calculate the loss (error) between prediction and target
2. Compute gradients—how much each weight contributed to the error
3. Adjust weights in the opposite direction of their gradients to reduce future errors

This is the core learning mechanism. Through thousands or millions of examples, the network's weights are gradually adjusted to minimize prediction errors.

**What are "learned representations"?** The patterns encoded in the network's weights. After training, the hidden layers have learned to recognize meaningful features:
- Early layers: edges, colors, textures
- Middle layers: shapes, patterns, combinations
- Later layers: objects, concepts, high-level features

### Step 4: Update and Repeat

The network repeatedly processes training examples, makes predictions, calculates errors, and updates its weights. This cycle continues until:

- Performance stops improving (convergence)
- A maximum number of training iterations is reached
- Performance on validation data starts degrading (overfitting)

**Training dynamics:**
- **Early in training:** Large errors, rapid improvement as basic patterns are learned
- **Mid training:** Errors decrease more slowly as the network refines its understanding
- **Late training:** Incremental improvements, risk of overfitting if trained too long

## Putting It Together: A Complete Example

Let's trace how a neural network learns to recognize handwritten digits (the classic MNIST problem):

**Setup:**
- Input layer: 784 neurons (28×28 pixel grayscale images flattened)
- Hidden layers: 2 layers with 128 and 64 neurons
- Output layer: 10 neurons (digits 0-9)
- Training data: 60,000 labeled digit images

**Training process:**

1. **Initialize:** All weights are random—the network has no idea what a digit looks like

2. **Show example #1:** An image of the digit "7"
   - Forward pass through layers → Output predicts "3" (wrong!)
   - Calculate loss: high error
   - Backpropagate: adjust weights to do better on "7"

3. **Show example #2:** An image of the digit "1"
   - Forward pass → Output predicts "1" (correct!)
   - Calculate loss: low error (but weights still need refinement)

4. **Show examples #3-60,000:** Repeat the process...

5. **After training:** The network has learned:
   - Early hidden layers: recognize strokes, loops, line crossings
   - Later hidden layers: combine strokes into digit-like patterns
   - Output layer: maps patterns to digit labels

6. **Testing:** Show the network a new "7" it's never seen:
   - Forward pass → Output predicts "7" with 98% confidence!

## Information Flow

Data flows through a neural network in one direction during prediction (forward pass) and gradients flow backward during learning (backward pass).

### Training Mode: Forward Pass + Backward Pass

```
FORWARD PASS (compute prediction)
══════════════════════════════════════

Input Layer
    ↓
Hidden Layers (transform data)
    ↓
Output Layer → Prediction
    ↓
Compare with Target → Compute Loss


BACKWARD PASS (compute gradients & update weights)
══════════════════════════════════════════════════

Loss → Calculate gradients
    ↓
Update Output Layer weights
    ↓
Update Hidden Layer weights
    ↓
Input Layer (no weights to update)
```

### Inference Mode: Forward Pass Only

```
Input Layer
    ↓
Hidden Layers (apply learned transformations)
    ↓
Output Layer → Prediction

(No loss, no gradients, no weight updates)
```

### Key Difference

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Forward pass** | ✓ Yes | ✓ Yes |
| **Loss computed** | ✓ Yes | ✗ No |
| **Gradients calculated** | ✓ Yes | ✗ No |
| **Weights updated** | ✓ Yes | ✗ No |

**Inference (prediction):** Once trained, the network only performs forward passes to make predictions on new data. No learning occurs—weights are fixed.

## Key Takeaways

| Component | Function |
|-----------|----------|
| **Input Layer** | Receives numerically encoded data as tensors |
| **Hidden Layers** | Learn hierarchical patterns through linear + non-linear transformations |
| **Output Layer** | Produces predictions or learned representations |
| **Forward Pass** | Data flows through layers to produce predictions |
| **Backpropagation** | Gradients flow backward to update weights based on errors |
| **Weights** | Learned parameters that encode the network's knowledge |

## Discussion Questions

1. **Why do we need hidden layers?** Could a neural network with just an input and output layer learn anything useful? What would it be limited to?

2. **Deeper networks can learn more complex patterns, but they also require more data and computation.** How might you decide how many hidden layers to use for a given problem?

3. **The network learns representations in hidden layers automatically.** How is this different from traditional machine learning where you manually engineer features?

4. **Consider the digit recognition example.** What do you think the first hidden layer neurons might be detecting? What about the second hidden layer?

## Preview: How PyTorch Enables This

Now that you understand the conceptual anatomy of neural networks, you're probably wondering: *How do we actually build these in practice?*

PyTorch provides the tools to:
- Create tensors from your data
- Define neural network architectures
- Perform forward passes
- Calculate losses
- Update weights through backpropagation

In the next sections, you'll learn the PyTorch essentials that make building neural networks practical and efficient.

## Next Steps

- [Learning Methodology](03-learning-methodology.md) - Strategies for effective deep learning study
- [PyTorch Essentials](04-pytorch-essentials.md) - The framework that makes all this possible

---

**Last Updated**: January 2026
