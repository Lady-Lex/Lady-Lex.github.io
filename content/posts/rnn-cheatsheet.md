---
title: "RNN Coding Cheatsheet"
date: 2025-11-06T14:00:00+08:00
tags: ["RNN", "LSTM", "GRU", "Deep Learning", "PyTorch", "Sequence", "Cheatsheet", "Exam"]
categories: ["CEG5304"]
draft: false
summary: "Complete RNN/LSTM/GRU implementations including sequence models, attention mechanism, and time series prediction for lab exam preparation."
---

# RNN Coding Cheatsheet
## EE5934/6934 Lab Exam Preparation

---

## 🔄 Recurrent Neural Networks (RNN)

### Basic RNN Cell (NumPy)
```python
import numpy as np

class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((1, hidden_size))
    
    def forward(self, x, h_prev):
        """
        x: input at current timestep (batch_size, input_size)
        h_prev: hidden state from previous timestep (batch_size, hidden_size)
        Returns: h_next (batch_size, hidden_size)
        """
        h_next = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
        return h_next

# Usage
rnn_cell = RNNCell(input_size=10, hidden_size=20)
x = np.random.randn(1, 10)  # Single input
h = np.zeros((1, 20))  # Initial hidden state

# Process sequence
for t in range(5):  # 5 timesteps
    h = rnn_cell.forward(x, h)
    print(f"Timestep {t}: hidden state shape {h.shape}")
```

### Basic RNN (PyTorch)
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        x: (batch_size, sequence_length, input_size)
        h0: (num_layers, batch_size, hidden_size) - optional initial hidden state
        """
        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # RNN forward pass
        # out: (batch_size, seq_len, hidden_size)
        # hn: (num_layers, batch_size, hidden_size)
        out, hn = self.rnn(x, h0)
        
        # Use output from last timestep
        out = self.fc(out[:, -1, :])  # (batch_size, output_size)
        
        return out, hn

# Usage
model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
x = torch.randn(32, 15, 10)  # (batch=32, seq_len=15, features=10)
output, hidden = model(x)
print(f"Output shape: {output.shape}")  # [32, 5]
print(f"Hidden shape: {hidden.shape}")  # [1, 32, 20]
```

### LSTM Implementation
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        x: (batch_size, sequence_length, input_size)
        hidden: tuple of (h0, c0) or None
        """
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            hidden = (h0, c0)
        
        # LSTM forward
        # out: (batch_size, seq_len, hidden_size)
        # (hn, cn): each (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x, hidden)
        
        # Use last timestep
        out = self.fc(out[:, -1, :])
        
        return out, (hn, cn)

# Usage
model = LSTMModel(input_size=10, hidden_size=50, output_size=2, num_layers=2)
x = torch.randn(16, 20, 10)  # (batch=16, seq_len=20, features=10)
output, (h, c) = model(x)
print(f"Output: {output.shape}")  # [16, 2]
```

### GRU Implementation
```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        x: (batch_size, sequence_length, input_size)
        """
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        
        return out, hn

# Usage
model = GRUModel(input_size=10, hidden_size=30, output_size=5)
```

### Bidirectional RNN
```python
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # KEY: bidirectional
        )
        
        # Output layer: hidden_size * 2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # out: (batch, seq_len, hidden_size * 2)
        out, _ = self.lstm(x)
        
        # Use last timestep (forward + backward)
        out = self.fc(out[:, -1, :])
        
        return out

# Usage
model = BiRNN(input_size=10, hidden_size=20, output_size=3)
x = torch.randn(8, 15, 10)
output = model(x)
print(f"Output shape: {output.shape}")  # [8, 3]
```

---

## 📐 RNN Architectures

### Many-to-One RNN (Sequence Classification)
```python
class Seq2One(nn.Module):
    """
    Input: sequence of vectors
    Output: single vector (e.g., sentiment classification)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2One, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        
        # Use final hidden state
        out = self.fc(hn[-1])  # (batch, output_size)
        
        return out
```

### Many-to-Many RNN (Sequence to Sequence)
```python
class Seq2Seq(nn.Module):
    """
    Input: sequence of vectors
    Output: sequence of vectors (e.g., time series prediction)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # out: (batch, seq_len, hidden_size)
        
        # Apply FC to each timestep
        out = self.fc(out)  # (batch, seq_len, output_size)
        
        return out
```

### One-to-Many RNN (e.g., Image Captioning)
```python
class One2Many(nn.Module):
    """
    Input: single vector
    Output: sequence of vectors
    """
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(One2Many, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # Encode input
        self.encoder = nn.Linear(input_size, hidden_size)
        
        # LSTM decoder
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state from input
        h = torch.tanh(self.encoder(x))
        c = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        
        # Generate sequence
        for t in range(self.seq_len):
            h, c = self.lstm(h, (h, c))
            out = self.fc(h)
            outputs.append(out)
        
        # Stack outputs: (batch, seq_len, output_size)
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
```

### RNN for Text Processing
```python
class TextRNN(nn.Module):
    """Text classification with RNN"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(TextRNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len) - indices
        
        # Embed: (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(embedded)
        
        # Use last hidden state
        out = self.fc(hn[-1])
        
        return out

# Usage
vocab_size = 10000
model = TextRNN(vocab_size, embedding_dim=100, hidden_size=128, output_size=2)
# Input: batch of sequences (word indices)
x = torch.randint(0, vocab_size, (16, 50))  # (batch=16, seq_len=50)
output = model(x)
print(f"Output shape: {output.shape}")  # [16, 2]
```

---

## 🔧 Manual Implementations

### Manual RNN Forward Pass
```python
def rnn_forward_manual(x, W_xh, W_hh, b_h, h0):
    """
    Manual implementation of RNN forward pass
    x: (seq_len, input_size)
    W_xh: (input_size, hidden_size)
    W_hh: (hidden_size, hidden_size)
    b_h: (hidden_size,)
    h0: (hidden_size,) - initial hidden state
    """
    seq_len = x.shape[0]
    hidden_size = W_hh.shape[0]
    
    # Store hidden states
    hidden_states = np.zeros((seq_len + 1, hidden_size))
    hidden_states[0] = h0
    
    # Process each timestep
    for t in range(seq_len):
        # h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        hidden_states[t + 1] = np.tanh(
            np.dot(x[t], W_xh) + 
            np.dot(hidden_states[t], W_hh) + 
            b_h
        )
    
    return hidden_states[1:]  # Return all hidden states except h0

# Example
seq_len, input_size, hidden_size = 10, 5, 20
x = np.random.randn(seq_len, input_size)
W_xh = np.random.randn(input_size, hidden_size) * 0.01
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
b_h = np.zeros(hidden_size)
h0 = np.zeros(hidden_size)

hidden_states = rnn_forward_manual(x, W_xh, W_hh, b_h, h0)
print(f"Hidden states shape: {hidden_states.shape}")  # (10, 20)
```

### LSTM Cell Manual Implementation
```python
class LSTMCell:
    """Manual LSTM cell implementation"""
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Forget gate
        self.W_f = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros(hidden_size)
        
        # Input gate
        self.W_i = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros(hidden_size)
        
        # Candidate cell state
        self.W_c = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros(hidden_size)
        
        # Output gate
        self.W_o = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x, h_prev, c_prev):
        """
        LSTM forward pass
        x: input at current timestep
        h_prev: previous hidden state
        c_prev: previous cell state
        """
        # Concatenate input and hidden state
        combined = np.concatenate([x, h_prev], axis=0)
        
        # Forget gate
        f_t = self.sigmoid(np.dot(combined, self.W_f) + self.b_f)
        
        # Input gate
        i_t = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        
        # Candidate cell state
        c_tilde = np.tanh(np.dot(combined, self.W_c) + self.b_c)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate
        o_t = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        
        # Update hidden state
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t

# Usage
lstm_cell = LSTMCell(input_size=10, hidden_size=20)
x = np.random.randn(10)
h = np.zeros(20)
c = np.zeros(20)

h_next, c_next = lstm_cell.forward(x, h, c)
print(f"h shape: {h_next.shape}, c shape: {c_next.shape}")
```

---

## 📊 Training RNN Models

### Training RNN for Sequence Classification
```python
def train_rnn_classifier(model, train_loader, criterion, optimizer, num_epochs):
    """
    Training loop for RNN-based sequence classifier
    """
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for sequences, labels in train_loader:
            # sequences: (batch, seq_len, input_size)
            # labels: (batch,)
            
            # Forward pass
            outputs, _ = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (important for RNN!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')

# Usage
model = SimpleRNN(input_size=10, hidden_size=50, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Time Series Prediction with RNN
```python
class TimeSeriesRNN(nn.Module):
    """RNN for time series forecasting"""
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(TimeSeriesRNN, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, 1)  # Predict single value
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Predict next timestep
        out = self.fc(out[:, -1, :])  # Use last timestep
        return out

# Create sequences for time series
def create_sequences(data, seq_length):
    """
    Create sequences from time series data
    data: 1D array of time series values
    seq_length: length of input sequence
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Usage
time_series = np.sin(np.linspace(0, 100, 1000))
X, y = create_sequences(time_series, seq_length=10)
X = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
y = torch.FloatTensor(y).unsqueeze(-1)

model = TimeSeriesRNN(input_size=1, hidden_size=50, num_layers=2)
```

---

## 🎯 Attention Mechanism

### Attention RNN
```python
class AttentionRNN(nn.Module):
    """RNN with attention mechanism"""
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Encoder
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention weights
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # Get all hidden states
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Calculate attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), 
            dim=1
        )  # (batch, seq_len)
        
        # Apply attention to get context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_out  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        # Output
        out = self.fc(context)
        
        return out, attention_weights

# Usage
model = AttentionRNN(input_size=10, hidden_size=50, output_size=2)
x = torch.randn(8, 15, 10)
output, attention = model(x)
print(f"Output: {output.shape}, Attention: {attention.shape}")
```

---

## 📐 RNN Dimension Reference

```python
"""
RNN/LSTM/GRU Input/Output Shapes:

Input: (batch_size, sequence_length, input_size)
Output: (batch_size, sequence_length, hidden_size)
Hidden: (num_layers, batch_size, hidden_size)

For LSTM, also have cell state (same shape as hidden):
(h_n, c_n): both (num_layers, batch_size, hidden_size)

For Bidirectional:
Output: (batch_size, sequence_length, hidden_size * 2)
Hidden: (num_layers * 2, batch_size, hidden_size)

batch_first=True means first dimension is batch
batch_first=False means (seq_len, batch, features)
"""
```

---

## ⚠️ Common RNN Pitfalls

```python
# ❌ WRONG: Not detaching hidden state between batches
for data, target in dataloader:
    output, hidden = model(data, hidden)  # hidden accumulates gradients!

# ✅ CORRECT: Detach hidden state
for data, target in dataloader:
    if hidden is not None:
        hidden = hidden.detach()
    output, hidden = model(data, hidden)

# ❌ WRONG: Vanishing/exploding gradients
# ✅ CORRECT: Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# ❌ WRONG: Wrong sequence dimension
x = torch.randn(10, 32, 50)  # (seq, batch, features) but batch_first=True

# ✅ CORRECT: Match batch_first setting
x = torch.randn(32, 10, 50)  # (batch, seq, features) when batch_first=True
```

---

## ⚡ Quick Tips

### Key Concepts
- **Hidden State**: Carries information across timesteps
- **LSTM**: Uses cell state to combat vanishing gradients
- **GRU**: Simplified LSTM with fewer parameters
- **Bidirectional**: Processes sequence in both directions

### Common Settings
- **Learning Rate**: Start with 0.001
- **Gradient Clipping**: max_norm=5.0 for stability
- **Dropout**: 0.2-0.5 between layers
- **Hidden Size**: 50-512 depending on task complexity

### Training Tips
- Always use gradient clipping for RNNs
- Detach hidden states between batches
- Use LSTM/GRU instead of vanilla RNN for better performance
- Consider bidirectional for sequence classification

---

## 📚 Import Statements

```python
# Essential imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# For visualization
import matplotlib.pyplot as plt
```

---

**Good luck with your exam! 🎓**

*Remember: RNNs are all about sequence! Think about how information flows through time.*

