---
title: "Autoencoder Coding Cheatsheet"
date: 2025-11-06T15:00:00+08:00
tags: ["Autoencoder", "Deep Learning", "PyTorch", "Denoising", "Cheatsheet", "Exam"]
categories: ["CEG5304"]
draft: false
summary: "Complete autoencoder implementations including basic AE, denoising AE, stacked AE, and convolutional AE for EE5934/6934 lab exam preparation."
---

# Autoencoder Coding Cheatsheet
## EE5934/6934 Lab Exam Preparation

---

## 🔧 Autoencoder Implementations

### Basic Autoencoder
```python
import numpy as np
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # or Tanh for normalized data
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Usage
model = Autoencoder(input_dim=784, hidden_dim=128)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in dataloader:
        # Forward pass
        output = model(data)
        loss = criterion(output, data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Denoising Autoencoder
```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_factor=0.3):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_factor = noise_factor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = x + noise
        return torch.clamp(noisy_x, 0., 1.)
    
    def forward(self, x, add_noise=True):
        if add_noise:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded

# Training
model = DenoisingAutoencoder(784, 128)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    for clean_data in dataloader:
        # Forward with noise
        reconstructed = model(clean_data, add_noise=True)
        # Loss against clean data
        loss = criterion(reconstructed, clean_data)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Marginalized Denoising Autoencoder
```python
def add_dropout_noise(x, dropout_prob=0.5):
    """Set each feature to 0 with probability p"""
    mask = torch.bernoulli(torch.ones_like(x) * (1 - dropout_prob))
    return x * mask

class MarginalizedDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.5):
        super(MarginalizedDAE, self).__init__()
        self.dropout_prob = dropout_prob
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Apply dropout noise
        x_corrupted = add_dropout_noise(x, self.dropout_prob)
        encoded = self.encoder(x_corrupted)
        decoded = self.decoder(encoded)
        return decoded
```

### Stacked Autoencoder
```python
class StackedAutoencoder(nn.Module):
    def __init__(self, layer_dims):
        """
        layer_dims: list of dimensions [input, hidden1, hidden2, ..., code]
        Example: [784, 512, 256, 128]
        """
        super(StackedAutoencoder, self).__init__()
        
        # Build encoder
        encoder_layers = []
        for i in range(len(layer_dims) - 1):
            encoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        for i in range(len(layer_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i-1]))
            if i > 1:  # No activation on last layer
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Usage
model = StackedAutoencoder([784, 512, 256, 128])
```

---

## 📊 Training Loop for Autoencoders

### Complete Training Function
```python
def train_autoencoder(model, train_loader, criterion, optimizer, num_epochs, device='cpu'):
    """
    Complete training loop for autoencoder
    """
    model.to(device)
    model.train()
    
    training_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            # Get data (autoencoders don't need labels)
            if isinstance(data, (list, tuple)):
                data = data[0]  # If dataloader returns (data, labels)
            
            data = data.to(device)
            
            # Flatten if needed (for FC autoencoder)
            data = data.view(data.size(0), -1)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return training_losses

# Usage
model = Autoencoder(input_dim=784, hidden_dim=128)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = train_autoencoder(model, train_loader, criterion, optimizer, num_epochs=100)
```

### Validation Function
```python
def validate_autoencoder(model, val_loader, criterion, device='cpu'):
    """
    Evaluate autoencoder on validation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in val_loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            
            data = data.to(device)
            data = data.view(data.size(0), -1)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f'Validation Loss: {avg_loss:.6f}')
    
    return avg_loss
```

---

## 🎨 Visualization and Analysis

### Visualize Reconstructions
```python
import matplotlib.pyplot as plt

def visualize_reconstructions(model, data_loader, n_samples=10, device='cpu'):
    """
    Visualize original vs reconstructed images
    """
    model.eval()
    
    # Get a batch of data
    data = next(iter(data_loader))
    if isinstance(data, (list, tuple)):
        data = data[0]
    
    data = data[:n_samples].to(device)
    original = data.view(data.size(0), -1)
    
    # Get reconstructions
    with torch.no_grad():
        reconstructed = model(original)
    
    # Convert to numpy for visualization
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    plt.show()
```

### Visualize Latent Space
```python
def visualize_latent_space(model, data_loader, labels, device='cpu'):
    """
    Visualize 2D latent space (for autoencoder with 2D bottleneck)
    """
    model.eval()
    
    latent_vectors = []
    label_list = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            
            # Get encoded representation
            encoded = model.encoder(data)
            latent_vectors.append(encoded.cpu().numpy())
            label_list.append(target.numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                         c=label_list, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.show()
```

---

## 🔍 Feature Extraction with Autoencoders

### Extract Learned Features
```python
def extract_features(model, data_loader, device='cpu'):
    """
    Extract learned features (encoded representations) from autoencoder
    """
    model.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            
            # Get encoded features
            encoded = model.encoder(data)
            
            features.append(encoded.cpu().numpy())
            labels.append(target.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return features, labels

# Usage: Use extracted features for classification
features_train, labels_train = extract_features(model, train_loader)
features_test, labels_test = extract_features(model, test_loader)

# Train a classifier on extracted features
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(features_train, labels_train)
accuracy = clf.score(features_test, labels_test)
print(f'Classification accuracy: {accuracy:.4f}')
```

---

## 🛠️ Convolutional Autoencoder

### Conv Autoencoder for Images
```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)  # 7x7 -> 1x1
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),  # 1x1 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Usage
model = ConvAutoencoder()
input_img = torch.randn(1, 1, 28, 28)  # Batch=1, Channels=1, H=28, W=28
output = model(input_img)
print(f"Input shape: {input_img.shape}, Output shape: {output.shape}")
```

---

## 📋 Common Autoencoder Variants

### Variational Autoencoder (VAE) - Basic Structure
```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE Loss Function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
```

---

## ⚡ Quick Tips

### Key Concepts
- **Autoencoder Goal**: Learn compressed representation of data
- **Bottleneck**: The smallest hidden layer (latent representation)
- **Reconstruction Loss**: Usually MSE or Binary Cross-Entropy
- **Denoising**: Train on corrupted input, reconstruct clean output

### Common Architectures
- **Input Dim > Hidden Dim**: Compression/dimensionality reduction
- **Symmetric Architecture**: Encoder and decoder mirror each other
- **Activation**: ReLU for hidden layers, Sigmoid/Tanh for output

### Training Tips
- Start with small learning rate (0.001 or 0.0001)
- Use MSE loss for continuous values
- Monitor reconstruction quality visually
- Add regularization if overfitting occurs

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

# For datasets
from torchvision import transforms, datasets
```

---

**Good luck with your exam! 🎓**

