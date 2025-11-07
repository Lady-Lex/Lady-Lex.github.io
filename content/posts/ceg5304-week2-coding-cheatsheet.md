---
title: "CEG5304 Week 2 Coding Cheatsheet"
date: 2025-11-06T10:00:00+08:00
weight: 300
tags: ["CEG5304", "Week2", "KNN", "SVM", "Softmax", "Cheatsheet", "Coding", "Exam", "NumPy"]
categories: ["CEG5304"]
draft: false
summary: "K-Nearest Neighbors, SVM hinge loss, Softmax cross-entropy, distance metrics, and linear classifiers with gradient computation using NumPy."
---

> Copy-paste sections as needed. All code is **pure NumPy**, no torch.

---

## 0) Imports & Typing Stubs

```python
import numpy as np
from typing import Tuple, Iterable, Dict
```

---

## 1) Distance Metrics

**Formulas**

- Manhattan: $d_1(\mathbf{x},\mathbf{y})=\sum_i |x_i-y_i|$
- Euclidean: $d_2(\mathbf{x},\mathbf{y})=\sqrt{\sum_i (x_i-y_i)^2}$
- Minkowski: $d_p(\mathbf{x},\mathbf{y})=\left(\sum_i |x_i-y_i|^p\right)^{1/p}$

**Vectorized pairwise distances (Euclidean, no sqrt for speed):**

```python
def pairwise_l2(X_test: np.ndarray, X_train: np.ndarray, sqrt: bool=False) -> np.ndarray:
    tt = np.sum(X_test**2, axis=1, keepdims=True)
    tr = np.sum(X_train**2, axis=1, keepdims=True).T
    d2 = tt + tr - 2 * (X_test @ X_train.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2) if sqrt else d2
```

**Manhattan (broadcasted, memory-heavier):**

```python
def pairwise_l1(X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(X_test[:, None, :] - X_train[None, :, :]), axis=2)
```

---

## 2) K-Nearest Neighbors (KNN)

```python
class KNN:
    def __init__(self, k: int = 5, metric: str = "l2"):
        self.k = k
        self.metric = metric
        self.Xtr = None
        self.ytr = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.Xtr = X
        self.ytr = y

    def _dists(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "l1":
            return pairwise_l1(X, self.Xtr)
        elif self.metric == "l2":
            return pairwise_l2(X, self.Xtr, sqrt=False)
        else:
            raise ValueError("metric must be 'l1' or 'l2'")

    def predict(self, X: np.ndarray) -> np.ndarray:
        d = self._dists(X)
        idx = np.argpartition(d, self.k, axis=1)[:, :self.k]
        votes = self.ytr[idx]
        preds = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=int(self.ytr.max()+1)).argmax(),
            axis=1, arr=votes
        )
        return preds
```

**Accuracy helper & simple validation split**

```python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))

def train_val_split(X, y, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = int(len(y)*val_ratio)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]
```

---

## 3) Cross-Validation (KNN hyperparameter search)

```python
def kfold_indices(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        tr  = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tr, val

def cv_knn(X: np.ndarray, y: np.ndarray, k_list: Iterable[int], nfold: int = 5,
           metric: str = "l2") -> Dict[int, float]:
    scores = {}
    for k in k_list:
        fold_acc = []
        for tr_idx, val_idx in kfold_indices(len(y), nfold):
            clf = KNN(k=k, metric=metric)
            clf.fit(X[tr_idx], y[tr_idx])
            pred = clf.predict(X[val_idx])
            fold_acc.append(accuracy(y[val_idx], pred))
        scores[k] = float(np.mean(fold_acc))
    return scores
```

---

## 4) Linear Scores

$$\mathbf{s} = f(\mathbf{x}; W) = \mathbf{x}^\top W$$

```python
def scores(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return X @ W
```

---

## 5) SVM (Hinge) Loss

$$L_i = \sum_{j\ne y_i} \max(0, s_{ij}-s_{i,y_i}+1)$$

```python
def svm_loss(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> float:
    S = X @ W
    correct = S[np.arange(X.shape[0]), y][:, None]
    margins = np.maximum(0.0, S - correct + 1.0)
    margins[np.arange(X.shape[0]), y] = 0.0
    data_loss = np.mean(np.sum(margins, axis=1))
    reg_loss = reg * np.sum(W * W)
    return data_loss + reg_loss
```

---

## 6) Softmax + Cross-Entropy (numerically stable)

$$p_{ij}=\frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}, \quad L_i=-\log p_{i,y_i}$$

```python
def softmax_ce_loss(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> float:
    S = X @ W
    S -= S.max(axis=1, keepdims=True)
    logZ = np.log(np.sum(np.exp(S), axis=1))
    ll = S[np.arange(X.shape[0]), y] - logZ
    data_loss = -np.mean(ll)
    reg_loss = reg * np.sum(W * W)
    return data_loss + reg_loss
```

---

## 7) Gradient Variants

```python
def svm_loss_grad(W, X, y, reg=0.0):
    N, C = X.shape[0], W.shape[1]
    S = X @ W
    correct = S[np.arange(N), y][:, None]
    margins = S - correct + 1.0
    margins[np.arange(N), y] = 0.0
    mask = (margins > 0).astype(float)
    row_sum = np.sum(mask, axis=1)
    mask[np.arange(N), y] = -row_sum
    dW = X.T @ mask / N + 2*reg*W
    loss = np.mean(np.maximum(0.0, margins)) + reg*np.sum(W*W)
    return loss, dW

def softmax_ce_grad(W, X, y, reg=0.0):
    N = X.shape[0]
    S = X @ W
    S -= S.max(axis=1, keepdims=True)
    P = np.exp(S); P /= np.sum(P, axis=1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(N), y])) + reg*np.sum(W*W)
    P[np.arange(N), y] -= 1.0
    dW = X.T @ P / N + 2*reg*W
    return loss, dW
```

---

## 8) Tiny Usage Examples

```python
Xtr, ytr, Xte, yte = np.random.randn(500, 64), np.random.randint(0, 10, 500),                      np.random.randn(100, 64), np.random.randint(0, 10, 100)
knn = KNN(k=7, metric="l2")
knn.fit(Xtr, ytr)
pred = knn.predict(Xte)
print("KNN acc:", accuracy(yte, pred))

W = 0.001*np.random.randn(64, 10)
print("SVM loss:", svm_loss(W, Xtr, ytr, reg=1e-4))
print("CE loss:", softmax_ce_loss(W, Xtr, ytr, reg=1e-4))
```

---

## 9) Complexity & Pitfalls

- KNN: Train O(1); Test O(NM)
- SVM: Remember set correct-class margin to 0
- Softmax: Use log-sum-exp for stability
- Regularization: `+ reg*np.sum(W*W)` and `+ 2*reg*W` in grad

---

## 10) Function Signatures

```python
pairwise_l2(X_test, X_train, sqrt=False)
pairwise_l1(X_test, X_train)
class KNN:
    def fit(self, X, y): ...
    def predict(self, X): ...
svm_loss(W, X, y, reg=0.0)
softmax_ce_loss(W, X, y, reg=0.0)
```
