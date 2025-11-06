---
title: "Transformer Coding Cheatsheet (NumPy Only)"
date: 2025-11-06T16:00:00+08:00
tags: ["Transformer", "Deep Learning", "NumPy", "Attention", "Cheatsheet", "Exam"]
categories: ["CEG5304"]
draft: false
summary: "Complete Transformer implementation with attention mechanisms, positional encoding, multi-head attention, and layer normalization using pure NumPy."
---

# Transformer Coding Cheatsheet (NumPy Only)
## Lab Exam Ready Reference - Lecture 1-6

---

## 1. Scaled Dot-Product Attention

### Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Implementation
```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: Query matrix (batch_size, seq_len, d_k)
        K: Key matrix (batch_size, seq_len, d_k)
        V: Value matrix (batch_size, seq_len, d_v)
        mask: Mask matrix (optional)
    Returns:
        output: Attention output
        attention_weights: Attention weights
    """
    d_k = Q.shape[-1]
    
    # Step 1: QK^T
    scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq_len, seq_len)
    
    # Step 2: Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    if mask is not None:
        scores = scores + (mask * -1e9)  # Add large negative value to masked positions
    
    # Step 4: Softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Step 5: Multiply by V
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def softmax(x, axis=-1):
    """Stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

---

## 2. Sinusoidal Positional Encoding

### Formula
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Implementation
```python
def positional_encoding(max_seq_len, d_model):
    """
    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension
    Returns:
        pos_encoding: Positional encoding matrix (max_seq_len, d_model)
    """
    pos_encoding = np.zeros((max_seq_len, d_model))
    
    position = np.arange(0, max_seq_len).reshape(-1, 1)  # (max_seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sin to even indices
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    
    # Apply cos to odd indices
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

# Usage
def add_positional_encoding(embeddings, max_seq_len):
    """Add positional encoding to word embeddings"""
    seq_len, d_model = embeddings.shape
    pos_enc = positional_encoding(max_seq_len, d_model)
    return embeddings + pos_enc[:seq_len, :]
```

---

## 3. Multi-Head Attention

### Implementation
```python
def multi_head_attention(Q, K, V, num_heads, mask=None):
    """
    Args:
        Q, K, V: (batch_size, seq_len, d_model)
        num_heads: Number of attention heads
        mask: Optional mask
    Returns:
        output: (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads  # Dimension per head
    
    # Initialize weight matrices (in practice, these would be learned)
    W_Q = np.random.randn(d_model, d_model) * 0.01
    W_K = np.random.randn(d_model, d_model) * 0.01
    W_V = np.random.randn(d_model, d_model) * 0.01
    W_O = np.random.randn(d_model, d_model) * 0.01
    
    # Linear projections
    Q_proj = np.matmul(Q, W_Q)  # (batch, seq_len, d_model)
    K_proj = np.matmul(K, W_K)
    V_proj = np.matmul(V, W_V)
    
    # Split into multiple heads
    # Reshape to (batch, num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Apply attention for each head
    attention_output = np.zeros((batch_size, num_heads, seq_len, d_k))
    for i in range(num_heads):
        Q_head = Q_heads[:, i, :, :]  # (batch, seq_len, d_k)
        K_head = K_heads[:, i, :, :]
        V_head = V_heads[:, i, :, :]
        
        output, _ = scaled_dot_product_attention(Q_head, K_head, V_head, mask)
        attention_output[:, i, :, :] = output
    
    # Concatenate heads
    # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
    concat = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # Final linear projection
    output = np.matmul(concat, W_O)
    
    return output
```

---

## 4. Layer Normalization

### Formula
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

### Implementation
```python
def layer_normalization(x, gamma=None, beta=None, eps=1e-6):
    """
    Args:
        x: Input tensor (batch_size, seq_len, d_model)
        gamma: Scale parameter (initialized to 1)
        beta: Shift parameter (initialized to 0)
        eps: Small constant for numerical stability
    Returns:
        normalized: Layer normalized output
    """
    # Compute mean and variance along the last dimension
    mean = np.mean(x, axis=-1, keepdims=True)  # (batch, seq_len, 1)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # Scale and shift
    d_model = x.shape[-1]
    if gamma is None:
        gamma = np.ones(d_model)
    if beta is None:
        beta = np.zeros(d_model)
    
    output = gamma * x_normalized + beta
    
    return output
```

---

## 5. Masked Self-Attention (for Decoder)

### Implementation
```python
def create_look_ahead_mask(seq_len):
    """
    Create mask to prevent attending to future positions
    Args:
        seq_len: Sequence length
    Returns:
        mask: Upper triangular matrix (seq_len, seq_len)
    """
    # Create upper triangular matrix with 1s above diagonal
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask  # 1s will be masked, 0s will be kept

def masked_self_attention(Q, K, V):
    """
    Self-attention with look-ahead mask for decoder
    Args:
        Q, K, V: Query, Key, Value matrices
    Returns:
        output: Masked attention output
    """
    seq_len = Q.shape[1]
    mask = create_look_ahead_mask(seq_len)
    
    # Add batch dimension to mask
    mask = np.expand_dims(mask, axis=0)  # (1, seq_len, seq_len)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
    
    return output, attention_weights
```

---

## 6. Residual Connection

### Implementation
```python
def residual_connection(x, sublayer_output):
    """
    Add residual connection: output = x + sublayer(x)
    Args:
        x: Original input
        sublayer_output: Output from sublayer (attention or FFN)
    Returns:
        output: x + sublayer_output
    """
    return x + sublayer_output

def residual_with_layer_norm(x, sublayer_output, gamma=None, beta=None):
    """
    Residual connection followed by layer normalization
    Args:
        x: Original input
        sublayer_output: Output from sublayer
    Returns:
        output: LayerNorm(x + sublayer_output)
    """
    residual_sum = x + sublayer_output
    return layer_normalization(residual_sum, gamma, beta)
```

---

## 7. Feed-Forward Network

### Formula
```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

### Implementation
```python
def feed_forward_network(x, d_ff, d_model):
    """
    Position-wise Feed-Forward Network
    Args:
        x: Input (batch_size, seq_len, d_model)
        d_ff: Hidden dimension (typically 4 * d_model)
        d_model: Model dimension
    Returns:
        output: FFN output (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, _ = x.shape
    
    # Initialize weights (in practice, these are learned)
    W1 = np.random.randn(d_model, d_ff) * 0.01
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.01
    b2 = np.zeros(d_model)
    
    # First linear transformation + ReLU
    hidden = np.maximum(0, np.matmul(x, W1) + b1)  # ReLU activation
    
    # Second linear transformation
    output = np.matmul(hidden, W2) + b2
    
    return output

def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)
```

---

## 8. Complete Encoder Layer

### Implementation
```python
def transformer_encoder_layer(x, num_heads=8, d_ff=2048):
    """
    Single Transformer encoder layer
    Args:
        x: Input (batch_size, seq_len, d_model)
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
    Returns:
        output: Encoder layer output
    """
    d_model = x.shape[-1]
    
    # 1. Multi-head self-attention
    attention_output = multi_head_attention(x, x, x, num_heads)
    
    # 2. Residual connection + Layer normalization
    x = residual_with_layer_norm(x, attention_output)
    
    # 3. Feed-forward network
    ffn_output = feed_forward_network(x, d_ff, d_model)
    
    # 4. Residual connection + Layer normalization
    output = residual_with_layer_norm(x, ffn_output)
    
    return output
```

---

## 9. Padding Mask

### Implementation
```python
def create_padding_mask(seq):
    """
    Create mask for padded positions (assumes padding token is 0)
    Args:
        seq: Input sequence (batch_size, seq_len)
    Returns:
        mask: Padding mask (batch_size, 1, 1, seq_len)
    """
    # Create mask where padding tokens (0) are marked as 1
    mask = (seq == 0).astype(np.float32)
    
    # Add extra dimensions for broadcasting
    # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
    return mask[:, np.newaxis, np.newaxis, :]
```

---

## 10. Complete Attention Example

### Full Working Example
```python
def complete_attention_example():
    """Complete example of attention mechanism"""
    
    # Setup
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    
    # Create sample input (e.g., word embeddings)
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Add positional encoding
    x_with_pos = add_positional_encoding(x, max_seq_len=10)
    
    print("Input shape:", x_with_pos.shape)
    
    # Apply self-attention
    Q = K = V = x_with_pos
    attention_output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("Attention output shape:", attention_output.shape)
    print("Attention weights shape:", weights.shape)
    
    # Apply multi-head attention
    multi_head_output = multi_head_attention(Q, K, V, num_heads)
    print("Multi-head output shape:", multi_head_output.shape)
    
    # Apply layer normalization
    normalized = layer_normalization(multi_head_output)
    print("Normalized shape:", normalized.shape)
    
    return attention_output, weights

# Run example
# output, weights = complete_attention_example()
```

---

## Quick Reference: Key Dimensions

| Component | Input Shape | Output Shape |
|-----------|-------------|--------------|
| Attention | (B, L, D) | (B, L, D) |
| Positional Encoding | (L, D) | (L, D) |
| Multi-Head Attention | (B, L, D) | (B, L, D) |
| Layer Norm | (B, L, D) | (B, L, D) |
| FFN | (B, L, D) | (B, L, D) |

**Legend:**
- B = batch_size
- L = sequence_length
- D = d_model (embedding dimension)

---

## Common Pitfalls to Avoid

1. **Forgetting to scale** by √d_k in attention
2. **Wrong axis for softmax** (should be last axis: axis=-1)
3. **Incorrect mask application** (add -1e9 before softmax, not after)
4. **Dimension mismatch** in multi-head attention reshape
5. **Wrong normalization axis** in LayerNorm (normalize over d_model)

---

## Exam Strategy Tips

1. **Start with simple functions first** (softmax, layer norm)
2. **Test with small matrices** (2x2 or 3x3) to verify logic
3. **Print shapes frequently** during debugging
4. **Remember:** No PyTorch/TensorFlow, only NumPy!
5. **Use internet for NumPy syntax**, but write logic yourself

---

## Time-Saving Shortcuts

```python
# Commonly used operations
def matmul(a, b): return np.matmul(a, b)
def transpose(a): return a.transpose()
def reshape(a, shape): return a.reshape(shape)

# Quick test function
def test_shape(tensor, expected_shape, name="tensor"):
    assert tensor.shape == expected_shape, f"{name} shape mismatch!"
    print(f"✓ {name} shape correct: {tensor.shape}")
```

---

**Good luck on your exam! 加油！🚀**