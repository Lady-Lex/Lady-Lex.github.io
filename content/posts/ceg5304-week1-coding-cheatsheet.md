---
title: "CEG5304 Week 1 Coding Cheatsheet"
date: 2025-11-06
tags: ["CEG5304", "Week1", "Cheatsheet", "NumPy", "Machine Learning", "Coding", "Exam"]
categories: ["CEG5304"]
draft: false
summary: "Machine learning fundamentals including data preprocessing, loss functions, linear regression, evaluation metrics, and feature engineering with NumPy."
---

# CEG5304 Lab Exam Coding Cheatsheet

## 1. 数据预处理 (Data Preprocessing)
Z-score 归一化

```python
import numpy as np

def z_score_normalize(data):
    """
    Z-score normalization: x_norm = (x - μ) / σ
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + 1e-8)  # 加小值防止除零
    return normalized_data, mean, std

# 使用示例
data = np.array([[1, 2], [3, 4], [5, 6]])
normalized, mu, sigma = z_score_normalize(data)
```

Min-Max 归一化

```python
def min_max_normalize(data, feature_range=(0, 1)):
    """
    Min-Max normalization: x_norm = (x - x_min) / (x_max - x_min)
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    
    # 归一化到 [0, 1]
    normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    
    # 缩放到指定范围
    if feature_range != (0, 1):
        a, b = feature_range
        normalized_data = a + (b - a) * normalized_data
    
    return normalized_data, min_val, max_val

# 使用示例
data = np.array([[1, 2], [3, 4], [5, 6]])
normalized, min_v, max_v = min_max_normalize(data)
```

One-Hot 编码

```python
def one_hot_encode(labels, num_classes=None):
    """
    One-hot encoding for categorical labels
    """
    if num_classes is None:
        num_classes = len(np.unique(labels))
    
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# 使用示例
labels = np.array([0, 1, 2, 1, 0])
one_hot = one_hot_encode(labels, num_classes=3)
# 输出: [[1,0,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0]]
```

异常值处理

```python
def remove_outliers_zscore(data, threshold=3):
    """
    使用 Z-score 方法移除异常值
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    mask = z_scores < threshold
    return data[mask]

def remove_outliers_iqr(data):
    """
    使用 IQR (Interquartile Range) 方法移除异常值
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (data >= lower_bound) & (data <= upper_bound)
    return data[mask]
```

2. 损失函数 (Loss Functions)
0-1 Loss (分类)

```python
def zero_one_loss(y_pred, y_true):
    """
    0-1 Loss: l(F(x), y) = I(F(x) ≠ y)
    """
    return np.mean(y_pred != y_true)

# 使用示例
y_pred = np.array([0, 1, 1, 0, 1])
y_true = np.array([0, 1, 0, 0, 1])
loss = zero_one_loss(y_pred, y_true)  # 0.2
```

L2 Loss / MSE (回归)

```python
def l2_loss(y_pred, y_true):
    """
    L2 Loss (MSE): L = (1/n) * Σ(y_pred - y_true)²
    """
    return np.mean((y_pred - y_true) ** 2)

def l2_loss_gradient(X, y_pred, y_true):
    """
    L2 Loss 的梯度 (对于线性模型 y = Xw)
    ∂L/∂w = (2/n) * X^T * (y_pred - y_true)
    """
    n = len(y_true)
    return (2 / n) * X.T @ (y_pred - y_true)

# 使用示例
y_pred = np.array([1.2, 2.3, 3.1])
y_true = np.array([1.0, 2.0, 3.0])
loss = l2_loss(y_pred, y_true)
```

MAE (Mean Absolute Error)

```python
def mae_loss(y_pred, y_true):
    """
    MAE: L = (1/n) * Σ|y_pred - y_true|
    """
    return np.mean(np.abs(y_pred - y_true))
```

Cross-Entropy Loss (分类)

```python
def cross_entropy_loss(y_pred, y_true):
    """
    Cross-Entropy Loss (假设 y_pred 是概率分布)
    L = -Σ y_true * log(y_pred)
    """
    epsilon = 1e-15  # 防止 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred))
```

3. 线性回归模型 (Linear Regression)
基础线性回归

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        """
        训练模型: y = Xw + b
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.n_iter):
            # 前向传播
            y_pred = self.predict(X)
            
            # 计算损失
            loss = l2_loss(y_pred, y)
            self.losses.append(loss)
            
            # 计算梯度
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        """
        预测: y = Xw + b
        """
        return X @ self.weights + self.bias

# 使用示例
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
```

带正则化的线性回归 (Ridge Regression)

```python
class RidgeRegression(LinearRegression):
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.1):
        super().__init__(learning_rate, n_iterations)
        self.alpha = alpha  # 正则化系数
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iter):
            y_pred = self.predict(X)
            
            # L2 loss + L2 regularization
            loss = l2_loss(y_pred, y) + self.alpha * np.sum(self.weights ** 2)
            self.losses.append(loss)
            
            # 梯度 (包含正则化项)
            dw = (1 / n_samples) * X.T @ (y_pred - y) + 2 * self.alpha * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
```

4. 训练与评估流程
数据集划分

```python
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    划分训练集和测试集
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# 使用示例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

完整训练流程

```python
def train_and_evaluate(X, y, model, test_size=0.2):
    """
    完整的训练和评估流程
    """
    # 1. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # 2. 数据预处理
    X_train_norm, mean, std = z_score_normalize(X_train)
    X_test_norm = (X_test - mean) / (std + 1e-8)
    
    # 3. 训练模型
    model.fit(X_train_norm, y_train)
    
    # 4. 评估
    y_train_pred = model.predict(X_train_norm)
    y_test_pred = model.predict(X_test_norm)
    
    train_loss = l2_loss(y_train_pred, y_train)
    test_loss = l2_loss(y_test_pred, y_test)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return model, train_loss, test_loss
```

K-Fold 交叉验证

```python
def k_fold_cross_validation(X, y, model, k=5):
    """
    K-Fold 交叉验证
    """
    n_samples = len(X)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    scores = []
    
    for i in range(k):
        # 划分验证集和训练集
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n_samples
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # 训练和评估
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = l2_loss(y_pred, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

5. 评估指标 (Metrics)
回归指标

```python
def r2_score(y_pred, y_true):
    """
    R² (决定系数)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def rmse(y_pred, y_true):
    """
    Root Mean Squared Error
    """
    return np.sqrt(l2_loss(y_pred, y_true))
```

分类指标

```python
def accuracy(y_pred, y_true):
    """
    准确率
    """
    return np.mean(y_pred == y_true)

def confusion_matrix(y_pred, y_true, num_classes):
    """
    混淆矩阵
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix

def precision_recall_f1(y_pred, y_true, pos_label=1):
    """
    精确率、召回率和 F1 分数 (二分类)
    """
    TP = np.sum((y_pred == pos_label) & (y_true == pos_label))
    FP = np.sum((y_pred == pos_label) & (y_true != pos_label))
    FN = np.sum((y_pred != pos_label) & (y_true == pos_label))
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1
```


6. 特征工程
多项式特征

```python
def polynomial_features(X, degree=2):
    """
    生成多项式特征
    例如: [x1, x2] -> [1, x1, x2, x1², x1*x2, x2²]
    """
    n_samples, n_features = X.shape
    features = [np.ones((n_samples, 1))]  # bias term
    
    for d in range(1, degree + 1):
        for i in range(n_features):
            features.append((X[:, i] ** d).reshape(-1, 1))
    
    return np.hstack(features)
```

特征标准化类

```python
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X * self.std + self.mean
```

7. 可视化
训练过程可视化

```python
import matplotlib.pyplot as plt

def plot_training_curve(losses, title="Training Loss"):
    """
    绘制训练损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_predictions(y_true, y_pred, title="Predictions vs True Values"):
    """
    绘制预测值 vs 真实值
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.grid(True)
    plt.show()
```

8. 常用工具函数
数据加载

```python
def load_csv_data(filepath, delimiter=',', has_header=True):
    """
    加载 CSV 数据
    """
    data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=int(has_header))
    X = data[:, :-1]  # 假设最后一列是标签
    y = data[:, -1]
    return X, y
```

Batch 生成器

```python
def batch_generator(X, y, batch_size=32, shuffle=True):
    """
    Mini-batch 生成器
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

# 使用示例
for X_batch, y_batch in batch_generator(X_train, y_train, batch_size=32):
    # 训练代码
    pass
```

9. 快速检查清单
考前必查

```python
# ✅ NumPy 基础操作
arr = np.array([1, 2, 3])
arr.shape, arr.mean(), arr.std(), arr.sum()

# ✅ 数组索引和切片
arr[0], arr[1:3], arr[[0, 2]], arr[arr > 1]

# ✅ 矩阵运算
A @ B  # 矩阵乘法
A * B  # 元素乘法
A.T    # 转置

# ✅ 常用函数
np.mean(), np.std(), np.min(), np.max()
np.argmax(), np.argmin()
np.where(), np.clip()
np.concatenate(), np.stack(), np.hstack(), np.vstack()
```

常见错误检查

```python
# ⚠️ 防止除零
result = numerator / (denominator + 1e-8)

# ⚠️ 防止 log(0)
log_val = np.log(np.clip(value, 1e-15, None))

# ⚠️ 检查维度
assert X.shape[0] == y.shape[0], "样本数不匹配"

# ⚠️ 随机种子
np.random.seed(42)
```