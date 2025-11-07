---
title: "CEG5304 Week 3 Coding Cheatsheet"
date: 2025-11-07T10:30:00+08:00
weight: 400
tags: ["CEG5304", "Week3", "Deep Learning", "MLP", "Perceptron", "Backpropagation", "Cheatsheet", "Coding", "Exam", "NumPy"]
categories: ["CEG5304"]
draft: false
summary: "Deep learning fundamentals including perceptron, MLP, activation functions, backpropagation, regularization, and training techniques with NumPy."
---

> Deep Learning fundamentals: Perceptron, MLP, Backpropagation, Regularization  
> All code is **pure NumPy**, no torch.

---

## 1. 基础导入和设置

```python
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子（可复现）
np.random.seed(42)

# 常用参数
learning_rate = 0.01
epochs = 1000
batch_size = 32
```

---

## 2. 激活函数 & 导数

### Sigmoid

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

### Tanh

```python
def tanh(z):
    return np.tanh(z)
    # 或: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_derivative(z):
    return 1 - np.tanh(z)**2
```

### ReLU

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

---

## 3. 感知机（Perceptron）

### 单个感知机

```python
class Perceptron:
    def __init__(self, input_size):
        # 初始化权重 (包括bias)
        self.weights = np.random.randn(input_size + 1) * 0.01
    
    def predict(self, X):
        # 添加bias项 (x0 = 1)
        X_bias = np.c_[np.ones(X.shape[0]), X]
        # 计算加权和
        z = np.dot(X_bias, self.weights)
        # 阶跃函数
        return (z > 0).astype(int)
    
    def fit(self, X, y, epochs=100, lr=0.01):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        for epoch in range(epochs):
            for i in range(len(X)):
                # 预测
                z = np.dot(X_bias[i], self.weights)
                y_pred = 1 if z > 0 else 0
                
                # 更新权重: w = w + (d - y) * x
                error = y[i] - y_pred
                self.weights += lr * error * X_bias[i]
        
        return self
```

### 感知机使用示例

```python
# AND gate
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, y, epochs=100)
predictions = perceptron.predict(X)
```

---

## 4. 多层感知机（MLP）

### 完整MLP实现

```python
class MLP:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list, e.g., [2, 4, 3, 1] 
                     (input, hidden1, hidden2, output)
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """前向传播"""
        self.activations = [X]  # 存储每层的激活值
        self.z_values = []       # 存储每层的加权和
        
        A = X
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.z_values.append(Z)
            
            # 最后一层用sigmoid，其他层用ReLU
            if i == len(self.weights) - 1:
                A = sigmoid(Z)
            else:
                A = relu(Z)
            
            self.activations.append(A)
        
        return A
    
    def backward(self, X, y):
        """反向传播"""
        m = X.shape[0]  # batch size
        
        # 初始化梯度
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # 输出层误差
        dA = self.activations[-1] - y  # 对于MSE loss
        
        # 从后往前传播
        for i in reversed(range(len(self.weights))):
            # 当前层的加权和
            Z = self.z_values[i]
            
            # 激活函数的导数
            if i == len(self.weights) - 1:
                dZ = dA * sigmoid_derivative(Z)
            else:
                dZ = dA * relu_derivative(Z)
            
            # 计算梯度
            dW[i] = np.dot(self.activations[i].T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
            # 传播到前一层
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
        
        return dW, db
    
    def update_weights(self, dW, db, lr):
        """更新权重"""
        for i in range(len(self.weights)):
            self.weights[i] -= lr * dW[i]
            self.biases[i] -= lr * db[i]
    
    def train(self, X, y, epochs=1000, lr=0.01, batch_size=32):
        """训练"""
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Mini-batch梯度下降
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                y_pred = self.forward(X_batch)
                
                # 计算损失
                loss = np.mean((y_pred - y_batch)**2)
                epoch_loss += loss
                
                # 反向传播
                dW, db = self.backward(X_batch, y_batch)
                
                # 更新权重
                self.update_weights(dW, db, lr)
            
            losses.append(epoch_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """预测"""
        return self.forward(X)
```

### MLP使用示例

```python
# XOR问题
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 创建网络: 2输入 -> 4隐藏 -> 1输出
mlp = MLP([2, 4, 1])
losses = mlp.train(X, y, epochs=5000, lr=0.1)

# 预测
predictions = mlp.predict(X)
print("Predictions:", predictions.round())
```

---

## 5. 损失函数

### Mean Squared Error (MSE)

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]
```

### Binary Cross-Entropy

```python
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15  # 防止log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
```

---

## 6. 梯度下降变体

### Batch Gradient Descent

```python
def batch_gd(X, y, weights, lr, epochs):
    for epoch in range(epochs):
        # 使用所有数据
        y_pred = forward(X, weights)
        loss = compute_loss(y, y_pred)
        
        # 计算梯度
        grads = compute_gradients(X, y, y_pred)
        
        # 更新权重
        weights -= lr * grads
    
    return weights
```

### Stochastic Gradient Descent

```python
def sgd(X, y, weights, lr, epochs):
    n = X.shape[0]
    
    for epoch in range(epochs):
        for i in range(n):
            # 单个样本
            Xi = X[i:i+1]
            yi = y[i:i+1]
            
            y_pred = forward(Xi, weights)
            grads = compute_gradients(Xi, yi, y_pred)
            
            # 更新权重
            weights -= lr * grads
    
    return weights
```

### Mini-batch Gradient Descent

```python
def mini_batch_gd(X, y, weights, lr, epochs, batch_size=32):
    n = X.shape[0]
    
    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 分批处理
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            y_pred = forward(X_batch, weights)
            grads = compute_gradients(X_batch, y_batch, y_pred)
            
            # 更新权重
            weights -= lr * grads
    
    return weights
```

---

## 7. 正则化技术

### Dropout

```python
def dropout(X, keep_prob=0.5, training=True):
    """
    X: 输入
    keep_prob: 保留概率 (0.5 表示dropout 50%)
    training: 是否在训练模式
    """
    if not training:
        return X
    
    # 生成mask
    mask = np.random.rand(*X.shape) < keep_prob
    
    # 应用mask并缩放
    return (X * mask) / keep_prob  # inverted dropout

# 使用示例
class MLPWithDropout(MLP):
    def forward(self, X, training=True, keep_prob=0.5):
        self.activations = [X]
        self.z_values = []
        
        A = X
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.z_values.append(Z)
            
            if i == len(self.weights) - 1:
                A = sigmoid(Z)
            else:
                A = relu(Z)
                # 在隐藏层应用dropout
                if training:
                    A = dropout(A, keep_prob, training=True)
            
            self.activations.append(A)
        
        return A
```

### Weight Decay (L2 Regularization)

```python
def compute_loss_with_l2(y_true, y_pred, weights, lambda_reg=0.01):
    """
    lambda_reg: 正则化系数
    """
    # 基础损失
    loss = mse_loss(y_true, y_pred)
    
    # L2正则化项
    l2_penalty = 0
    for w in weights:
        l2_penalty += np.sum(w**2)
    
    return loss + (lambda_reg / 2) * l2_penalty

# 梯度更新时加入权重衰减
def update_weights_with_decay(weights, grads, lr, lambda_reg=0.01):
    for i in range(len(weights)):
        # w = w - lr * (grad + lambda * w)
        weights[i] -= lr * (grads[i] + lambda_reg * weights[i])
    
    return weights
```

### Early Stopping

```python
def train_with_early_stopping(model, X_train, y_train, X_val, y_val, 
                               epochs=1000, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(epochs):
        # 训练
        model.train_one_epoch(X_train, y_train)
        
        # 验证
        val_loss = model.evaluate(X_val, y_val)
        
        # 检查是否改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.get_weights()  # 保存最佳权重
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.set_weights(best_weights)  # 恢复最佳权重
            break
    
    return model
```

---

## 8. 数据集划分

### Train/Validation/Test Split

```python
def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 1 - train_ratio - val_ratio
    """
    n = len(X)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])

# 使用示例
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
```

### K-Fold Cross-Validation

```python
def k_fold_split(X, y, k=5):
    """
    返回k个(train, val)索引对
    """
    n = len(X)
    indices = np.random.permutation(n)
    fold_size = n // k
    
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        folds.append((train_idx, val_idx))
    
    return folds

# 使用示例
def cross_validate(X, y, k=5):
    folds = k_fold_split(X, y, k)
    scores = []
    
    for train_idx, val_idx in folds:
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = MLP([X.shape[1], 4, 1])
        model.train(X_train, y_train, epochs=100)
        
        val_loss = mse_loss(y_val, model.predict(X_val))
        scores.append(val_loss)
    
    return np.mean(scores), np.std(scores)
```

---

## 9. 常用工具函数

### 数据标准化

```python
def standardize(X):
    """Z-score标准化"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8), mean, std

def normalize(X):
    """Min-Max归一化到[0,1]"""
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)
```

### One-Hot编码

```python
def one_hot_encode(y, num_classes):
    """
    y: shape (n,)
    return: shape (n, num_classes)
    """
    n = len(y)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1
    return one_hot
```

### 准确率计算

```python
def accuracy(y_true, y_pred):
    """
    分类准确率
    """
    return np.mean((y_pred > 0.5) == y_true)

def confusion_matrix(y_true, y_pred):
    """
    混淆矩阵
    """
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    return np.array([[TN, FP], [FN, TP]])
```

---

## 10. 快速测试模板

```python
# ===== 快速测试模板 =====

# 1. 准备数据
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])  # XOR

# 2. 标准化 (可选)
X_norm, mean, std = standardize(X)

# 3. 划分数据集
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

# 4. 创建模型
model = MLP([2, 4, 1])  # 2输入 -> 4隐藏 -> 1输出

# 5. 训练
losses = model.train(X_train, y_train, epochs=1000, lr=0.1, batch_size=2)

# 6. 评估
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

print("Training Accuracy:", accuracy(y_train, train_pred))
print("Validation Accuracy:", accuracy(y_val, val_pred))

# 7. 可视化
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

---

## 11. 考试注意事项

### ✅ 必须会的

- Sigmoid/ReLU/Tanh及其导数
- 感知机权重更新公式
- 前向传播（手写矩阵乘法）
- 反向传播（链式法则）
- Mini-batch采样
- MSE损失函数

### ⚠️ 常见错误

```python
# ❌ 忘记添加bias
Z = np.dot(X, W)  # 错误

# ✅ 正确
Z = np.dot(X, W) + b

# ❌ 维度不匹配
W = np.random.randn(4, 2)  # 错误 (应该是2x4)

# ✅ 检查维度
print(f"X: {X.shape}, W: {W.shape}, Z: {Z.shape}")

# ❌ Dropout时忘记缩放
A = A * mask  # 错误

# ✅ Inverted dropout
A = (A * mask) / keep_prob
```

### 🔍 调试技巧

```python
# 打印中间值
def forward_debug(X, weights):
    print(f"Input shape: {X.shape}")
    Z = np.dot(X, weights)
    print(f"Z shape: {Z.shape}")
    print(f"Z sample: {Z[:3]}")  # 查看前3个值
    return Z

# 检查梯度（数值梯度）
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        old_value = x[i]
        
        x[i] = old_value + h
        fxh1 = f(x)
        
        x[i] = old_value - h
        fxh2 = f(x)
        
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = old_value
    
    return grad
```

---
