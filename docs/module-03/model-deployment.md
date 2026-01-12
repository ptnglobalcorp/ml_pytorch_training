# Model Deployment

## Learning Objectives

By the end of this lesson, you will be able to:
- Prepare models for production deployment
- Export models to different formats
- Create inference pipelines
- Deploy models as REST APIs
- Optimize models for deployment

## Model Preparation

### Finalizing the Model

Before deployment, ensure your model is in the correct state:

```python
import torch
import torch.nn as nn

# Load the best model
model = CNN(num_classes=10)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))

# Set to evaluation mode
model.eval()

# Verify the model works
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

# Save in the recommended format
torch.save(model.state_dict(), 'production_model.pth')
```

### Creating Model Metadata

```python
import json
from datetime import datetime

def save_model_metadata(model, model_path, metadata):
    """Save model with metadata for production"""

    # Calculate model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_metadata = {
        'model_name': metadata.get('name', 'classifier'),
        'version': metadata.get('version', '1.0.0'),
        'created_at': datetime.now().isoformat(),
        'framework': 'pytorch',
        'pytorch_version': torch.__version__,
        'model_architecture': str(model),
        'input_shape': metadata.get('input_shape'),
        'output_shape': metadata.get('output_shape'),
        'num_classes': metadata.get('num_classes'),
        'class_names': metadata.get('class_names', []),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'performance_metrics': metadata.get('metrics', {}),
        'training_data_info': metadata.get('training_info', {}),
        'preprocessing': metadata.get('preprocessing', {}),
        'postprocessing': metadata.get('postprocessing', {})
    }

    # Save metadata
    metadata_path = model_path.replace('.pth', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)

    print(f"Model metadata saved to {metadata_path}")
    return model_metadata

# Usage
metadata = {
    'name': 'image_classifier',
    'version': '1.0.0',
    'input_shape': [1, 3, 224, 224],
    'output_shape': [1, 10],
    'num_classes': 10,
    'class_names': ['class_0', 'class_1', 'class_2', 'class_3', 'class_4',
                    'class_5', 'class_6', 'class_7', 'class_8', 'class_9'],
    'metrics': {
        'test_accuracy': 0.95,
        'test_precision': 0.94,
        'test_recall': 0.94,
        'test_f1': 0.94
    },
    'preprocessing': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'resize': [224, 224]
    }
}

save_model_metadata(model, 'production_model.pth', metadata)
```

## Model Export Formats

### TorchScript Export

TorchScript is PyTorch's recommended format for production:

```python
# Method 1: Tracing
model.eval()
example_input = torch.randn(1, 3, 224, 224)

traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')

# Verify traced model
loaded_traced = torch.jit.load('model_traced.pt')
output = loaded_traced(example_input)
print("Traced model works!")

# Method 2: Scripting (better for models with control flow)
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# Verify scripted model
loaded_scripted = torch.jit.load('model_scripted.pt')
output = loaded_scripted(example_input)
print("Scripted model works!")
```

### ONNX Export

Export to ONNX for cross-platform deployment:

```python
# Export to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,           # Store trained parameters
    opset_version=14,             # ONNX version
    do_constant_folding=True,     # Optimize constants
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Verify ONNX model
import onnx
import onnxruntime as ort

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Test with ONNX Runtime
ort_session = ort.InferenceSession("model.onnx")
outputs = ort_session.run(None, {'input': dummy_input.numpy()})
print(f"ONNX Runtime output shape: {outputs[0].shape}")
```

## Inference Pipeline

### Basic Inference Class

```python
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ModelInference:
    def __init__(self, model_path, metadata_path=None, device='cpu'):
        """
        Args:
            model_path: Path to model file
            metadata_path: Path to model metadata JSON
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)

        # Load metadata if provided
        self.metadata = None
        self.class_names = None
        self.preprocessing = None

        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.class_names = self.metadata.get('class_names', [])
                self.preprocessing = self.metadata.get('preprocessing', {})

    def _load_model(self, model_path):
        """Load model from file"""
        if model_path.endswith('.pt'):
            # TorchScript model
            model = torch.jit.load(model_path, map_location=self.device)
        else:
            # Regular PyTorch model
            # Note: You need to define the model class first
            model = CNN(num_classes=10)  # Adjust based on your model
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()

        model = model.to(self.device)
        return model

    def preprocess(self, image):
        """Preprocess input image"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # Default preprocessing if not specified
        if not self.preprocessing:
            self.preprocessing = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'resize': [224, 224]
            }

        transform = transforms.Compose([
            transforms.Resize(tuple(self.preprocessing['resize'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.preprocessing['mean'],
                std=self.preprocessing['std']
            )
        ])

        return transform(image).unsqueeze(0)

    def predict(self, input_data):
        """
        Make prediction

        Args:
            input_data: Can be image path, PIL Image, or preprocessed tensor

        Returns:
            Dictionary with predictions
        """
        # Preprocess
        if not isinstance(input_data, torch.Tensor):
            input_tensor = self.preprocess(input_data)
        else:
            input_tensor = input_data

        input_tensor = input_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            if isinstance(self.model, torch.jit.ScriptModule):
                outputs = self.model(input_tensor)
            else:
                outputs = self.model(input_tensor)

        # Process outputs
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        # Prepare result
        result = {
            'predicted_class': predicted.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy().flatten().tolist()
        }

        if self.class_names:
            result['predicted_label'] = self.class_names[result['predicted_class']]
            result['class_probabilities'] = {
                self.class_names[i]: prob
                for i, prob in enumerate(result['probabilities'])
            }

        return result

    def predict_batch(self, input_list):
        """Make predictions on a batch of inputs"""
        results = []
        for input_data in input_list:
            result = self.predict(input_data)
            results.append(result)
        return results

# Usage
inference = ModelInference(
    model_path='production_model.pth',
    metadata_path='production_model_metadata.json',
    device='cpu'
)

# Single prediction
result = inference.predict('test_image.jpg')
print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## REST API Deployment

### FastAPI Inference Server

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io

app = FastAPI(title="Model Inference API")

# Initialize model globally
model_inference = None

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    class_probabilities: dict

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_inference
    model_path = "production_model.pth"
    metadata_path = "production_model_metadata.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_inference = ModelInference(model_path, metadata_path, device)
    print("Model loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Model Inference API",
        "status": "running",
        "model_info": model_inference.metadata if model_inference.metadata else {}
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict image class

    Args:
        file: Uploaded image file

    Returns:
        Prediction results
    """
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Make prediction
        result = model_inference.predict(image)

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict on batch of images

    Args:
        files: List of uploaded image files

    Returns:
        List of prediction results
    """
    try:
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            images.append(image)

        results = model_inference.predict_batch(images)
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running the Server

```bash
# Install dependencies
pip install fastapi uvicorn python-multipart

# Run the server
python api.py

# Or with uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Testing the API

```python
import requests

# Test single prediction
url = "http://localhost:8000/predict"
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    print(response.json())

# Test batch prediction
url = "http://localhost:8000/predict_batch"
files = [("files", open(f"test_{i}.jpg", "rb")) for i in range(3)]
response = requests.post(url, files=files)
print(response.json())
```

## Model Optimization

### Quantization

Reduce model size and increase inference speed:

```python
# Dynamic quantization (post-training)
import torch.quantization

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'quantized_model.pth')

# Compare sizes
import os
original_size = os.path.getsize('production_model.pth')
quantized_size = os.path.getsize('quantized_model.pth')
print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
print(f"Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
print(f"Reduction: {(1 - quantized_size / original_size) * 100:.2f}%")
```

### Pruning

Remove less important weights:

```python
from torch.nn.utils import prune

# Prune model
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)  # Remove 20% of weights
        prune.remove(module, 'weight')  # Make pruning permanent

# Save pruned model
torch.save(model.state_dict(), 'pruned_model.pth')
```

## Deployment Best Practices

| Practice | Description |
|----------|-------------|
| **Model Versioning** | Track model versions and metadata |
| **Input Validation** | Validate inputs before inference |
| **Error Handling** | Handle errors gracefully |
| **Monitoring** | Track prediction latency and accuracy |
| **Security** | Authenticate API endpoints |
| **Batch Processing** | Support batch inference for efficiency |
| **Model Optimization** | Use quantization/pruning for production |

## Complete Deployment Checklist

```python
def deployment_checklist():
    """
    Checklist for model deployment
    """
    checklist = {
        "Model": [
            "✓ Model saved in correct format (.pth or .pt)",
            "✓ Model metadata documented",
            "✓ Model tested on test set",
            "✓ Model performance metrics recorded",
            "✓ Model optimized (quantized/pruned if needed)"
        ],
        "Inference": [
            "✓ Preprocessing pipeline defined",
            "✓ Postprocessing pipeline defined",
            "✓ Batch inference supported",
            "✓ Error handling implemented",
            "✓ Input validation added"
        ],
        "API": [
            "✓ API endpoints documented",
            "✓ Authentication implemented",
            "✓ Rate limiting configured",
            "✓ Monitoring/logging added",
            "✓ Health check endpoint"
        ],
        "Testing": [
            "✓ Unit tests for inference",
            "✓ Integration tests for API",
            "✓ Load testing performed",
            "✓ Edge cases tested"
        ]
    }

    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")

# Run checklist
deployment_checklist()
```

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Model Preparation** includes finalizing and metadata | Document all aspects of the model |
| **TorchScript** is PyTorch's production format | Use tracing or scripting for export |
| **ONNX** enables cross-platform deployment | Export for deployment on different platforms |
| **Inference Pipeline** wraps model for easy use | Handle preprocessing and postprocessing |
| **REST API** exposes model as service | Use FastAPI for easy API creation |
| **Optimization** reduces size and latency | Quantization and pruning for production |

## Practice Exercises

1. Create a complete inference pipeline for your classifier
2. Export your model to both TorchScript and ONNX formats
3. Build a FastAPI inference server with authentication
4. Implement model quantization and compare performance
5. Create a deployment checklist for your project

## Next Steps

- [Classification Basics](classification-basics.md) - Understanding classification tasks
- [Architecture Design](architecture-design.md) - Designing neural network architectures
- [Training & Evaluation](training-evaluation.md) - Training and evaluating classifiers

---

**Last Updated**: January 2026
