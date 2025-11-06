---
title: "CNN Coding Cheatsheet"
date: 2025-11-06T11:00:00+08:00
tags: ["CNN", "Deep Learning", "PyTorch", "Computer Vision", "Cheatsheet", "Exam"]
categories: ["CEG5304"]
draft: false
summary: "CNN implementation guide with convolution/pooling operations, complete architectures, training loops, and parameter calculations for lab exam."
---

# CNN Coding Cheatsheet
## EE5934/6934 Lab Exam Preparation

---

## 📋 Quick Reference: Output Size Formulas

### Convolution Layer Output Size
```
Output = (N - F + 2P) / S + 1

N = Input size (width or height)
F = Filter size
P = Padding
S = Stride
```

### Pooling Layer Output Size
```
Output = (W₁ - F) / S + 1

W₁ = Input size
F = Pool size
S = Stride
```

### Parameters Count
```
Conv Layer Parameters = (F × F × D_in + 1) × K

F = Filter size
D_in = Input depth
K = Number of filters
+1 = Bias term
```

---

## 🖼️ CNN Components

### Manual 2D Convolution (NumPy)
```python
import numpy as np

def conv2d(input_img, kernel, stride=1, padding=0):
    """
    Manual 2D convolution implementation
    input_img: (H, W) or (H, W, C)
    kernel: (F, F) or (F, F, C)
    """
    # Add padding
    if padding > 0:
        input_img = np.pad(input_img, 
                          ((padding, padding), (padding, padding)), 
                          mode='constant')
    
    H, W = input_img.shape[:2]
    F = kernel.shape[0]
    
    # Calculate output dimensions
    out_H = (H - F) // stride + 1
    out_W = (W - F) // stride + 1
    
    output = np.zeros((out_H, out_W))
    
    # Perform convolution
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            receptive_field = input_img[h_start:h_start+F, w_start:w_start+F]
            output[i, j] = np.sum(receptive_field * kernel)
    
    return output

# Example usage
input_img = np.random.rand(32, 32)
kernel = np.random.rand(3, 3)
output = conv2d(input_img, kernel, stride=1, padding=1)
print(f"Input shape: {input_img.shape}")
print(f"Output shape: {output.shape}")
```

### Multi-Channel Convolution
```python
def conv2d_multichannel(input_img, kernel, stride=1, padding=0):
    """
    input_img: (H, W, C_in)
    kernel: (F, F, C_in)
    """
    if padding > 0:
        input_img = np.pad(input_img,
                          ((padding, padding), (padding, padding), (0, 0)),
                          mode='constant')
    
    H, W, C_in = input_img.shape
    F = kernel.shape[0]
    
    out_H = (H - F) // stride + 1
    out_W = (W - F) // stride + 1
    
    output = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            receptive_field = input_img[h_start:h_start+F, w_start:w_start+F, :]
            # Sum across all channels
            output[i, j] = np.sum(receptive_field * kernel)
    
    return output
```

### Max Pooling
```python
def max_pool2d(input_img, pool_size=2, stride=2):
    """
    Max pooling implementation
    input_img: (H, W) or (H, W, C)
    """
    H, W = input_img.shape[:2]
    
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    if input_img.ndim == 3:
        C = input_img.shape[2]
        output = np.zeros((out_H, out_W, C))
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * stride
                    w_start = j * stride
                    pool_region = input_img[h_start:h_start+pool_size, 
                                           w_start:w_start+pool_size, c]
                    output[i, j, c] = np.max(pool_region)
    else:
        output = np.zeros((out_H, out_W))
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                w_start = j * stride
                pool_region = input_img[h_start:h_start+pool_size, 
                                       w_start:w_start+pool_size]
                output[i, j] = np.max(pool_region)
    
    return output

# Example
input_img = np.random.rand(28, 28)
pooled = max_pool2d(input_img, pool_size=2, stride=2)
print(f"Input: {input_img.shape}, Output: {pooled.shape}")
```

### Average Pooling
```python
def avg_pool2d(input_img, pool_size=2, stride=2):
    """Average pooling"""
    H, W = input_img.shape[:2]
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    output = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            pool_region = input_img[h_start:h_start+pool_size, 
                                   w_start:w_start+pool_size]
            output[i, j] = np.mean(pool_region)
    
    return output
```

---

## 🏗️ Complete CNN Architecture (PyTorch)

### Simple CNN for Classification
```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Conv Layer 1: 3x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, 
                               kernel_size=5, stride=1, padding=0)
        # Pool 1: 6x28x28 -> 6x14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, 
                               kernel_size=5, stride=1, padding=0)
        # Pool 2: 16x10x10 -> 16x5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool1(F.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Usage
model = SimpleCNN(num_classes=10)
input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # [1, 10]
```

### CNN with Batch Normalization
```python
class CNNWithBatchNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNWithBatchNorm, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2: 16x16 -> 8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3: 8x8 -> 4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and FC
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

---

## 🎯 Activation Functions

### Common Activations
```python
import torch
import torch.nn as nn

# ReLU
def relu(x):
    return np.maximum(0, x)

class ReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
def tanh(x):
    return np.tanh(x)

# Absolute Tanh
def abs_tanh(x):
    return np.abs(np.tanh(x))

# Quick reference in PyTorch
activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.01),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(dim=1)
}
```

---

## 📊 Training Loop Templates

### Basic Training Loop
```python
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

# Usage
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
```

### Validation Loop
```python
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
    
    return val_loss, val_acc
```

---

## 🛡️ Adversarial Training

### Generate Adversarial Examples (FGSM)
```python
def fgsm_attack(model, data, target, epsilon=0.1):
    """
    Fast Gradient Sign Method
    """
    # Require gradient
    data.requires_grad = True
    
    # Forward pass
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    
    # Backward pass to get gradient
    model.zero_grad()
    loss.backward()
    
    # Generate perturbation
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create adversarial example
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

# Adversarial Training Loop
def adversarial_training(model, train_loader, optimizer, epsilon=0.1):
    model.train()
    
    for data, target in train_loader:
        # Generate adversarial examples
        adv_data = fgsm_attack(model, data, target, epsilon)
        
        # Train on adversarial examples
        optimizer.zero_grad()
        output = model(adv_data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
```

---

## 🧮 Parameter Calculation Helper

### Calculate CNN Parameters
```python
def count_conv_params(in_channels, out_channels, kernel_size):
    """
    Calculate number of parameters in a Conv layer
    """
    params = (kernel_size * kernel_size * in_channels + 1) * out_channels
    return params

def count_fc_params(in_features, out_features):
    """
    Calculate number of parameters in a FC layer
    """
    params = (in_features + 1) * out_features
    return params

# Example
conv1_params = count_conv_params(3, 64, 3)
print(f"Conv1 parameters: {conv1_params}")  # (3*3*3 + 1) * 64 = 1,792

fc_params = count_fc_params(512, 10)
print(f"FC parameters: {fc_params}")  # (512 + 1) * 10 = 5,130

# Total model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = SimpleCNN()
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params:,}")
```

### Calculate Output Dimensions
```python
def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """Calculate output size after convolution"""
    return (input_size - kernel_size + 2 * padding) // stride + 1

def pool_output_size(input_size, pool_size, stride=None):
    """Calculate output size after pooling"""
    if stride is None:
        stride = pool_size
    return (input_size - pool_size) // stride + 1

# Example calculation
h, w = 32, 32
print(f"Input: {h}x{w}")

# After Conv(3x3, stride=1, pad=1)
h = conv_output_size(h, 3, 1, 1)
w = conv_output_size(w, 3, 1, 1)
print(f"After Conv: {h}x{w}")

# After MaxPool(2x2, stride=2)
h = pool_output_size(h, 2, 2)
w = pool_output_size(w, 2, 2)
print(f"After Pool: {h}x{w}")
```

---

## 🔍 Debugging Tools

### Check Shapes
```python
def check_model_shapes(model, input_size):
    """
    Print output shape of each layer
    """
    x = torch.randn(1, *input_size)
    print(f"Input shape: {x.shape}")
    
    for name, layer in model.named_children():
        x = layer(x)
        print(f"{name} output shape: {x.shape}")

# Usage
model = SimpleCNN()
check_model_shapes(model, (3, 32, 32))
```

### Gradient Checking
```python
def check_gradients(model):
    """Check if gradients are computed"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_mean={param.grad.mean():.6f}, grad_std={param.grad.std():.6f}")
        else:
            print(f"{name}: No gradient")

# After backward pass
loss.backward()
check_gradients(model)
```

---

## 📝 Common Exam Patterns

### Pattern 1: Implement from Scratch
```python
# Given: Input dimensions, architecture specification
# Task: Implement the network

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # TODO: Define layers based on specification
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass
```

### Pattern 2: Calculate Dimensions
```python
# Given: Network architecture
# Task: Calculate output dimensions at each layer

def calculate_dimensions(input_h, input_w, input_c):
    # Layer 1: Conv(F=5, S=1, P=0, K=6)
    h1 = (input_h - 5) // 1 + 1
    w1 = (input_w - 5) // 1 + 1
    c1 = 6
    print(f"After Conv1: {h1}x{w1}x{c1}")
    
    # Layer 2: MaxPool(F=2, S=2)
    h2 = (h1 - 2) // 2 + 1
    w2 = (w1 - 2) // 2 + 1
    c2 = c1
    print(f"After Pool1: {h2}x{w2}x{c2}")
    
    return h2, w2, c2
```

### Pattern 3: Fix Broken Code
```python
# Common errors to look for:
# 1. Wrong input/output dimensions
# 2. Missing activation functions
# 3. Incorrect loss function
# 4. Gradient not zeroed
# 5. Wrong reshape/view operations
```

---

## ⚡ Quick Tips

### Memory Tips
- **Conv output**: `(N - F + 2P) / S + 1`
- **Pool output**: `(W - F) / S + 1`
- **Parameters**: `(F × F × D_in + 1) × K`
- **Common settings**: F=3, S=1, P=1 (preserves size)

### Code Efficiency
```python
# Use view() or reshape() for flattening
x = x.view(batch_size, -1)  # -1 auto-calculates

# Use nn.Sequential for simple layers
self.features = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

# Save model checkpoints
torch.save(model.state_dict(), 'model.pth')
```

### Common Gotchas
- Always call `optimizer.zero_grad()` before backward
- Use `model.train()` and `model.eval()` appropriately
- Don't forget to move data to GPU if using: `data.to(device)`
- Check tensor shapes frequently during debugging

---

## 📚 Import Statements Reference

```python
# Essential imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# For visualization (if allowed)
import matplotlib.pyplot as plt

# Common utilities
from torchvision import transforms, datasets
```

---

**Good luck with your exam! 🎓**

*Remember: Focus on understanding concepts, not just memorizing code. The exam tests your ability to implement and debug, not just recall.*

