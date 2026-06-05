"""
Two-Layer Neural Network with Backpropagation (NumPy)
=====================================================
Architecture: Input(2) -> Hidden(4, sigmoid) -> Output(1, sigmoid)
Task: Learn the XOR function
"""

import numpy as np

# ── Activation & its derivative ──────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):
    """Derivative given the *activation* a = sigmoid(z)."""
    return a * (1.0 - a)

# ── Seed for reproducibility ────────────────────────────────────────
np.random.seed(42)

# ── XOR dataset ─────────────────────────────────────────────────────
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])          # (4, 2)

y = np.array([[0],
              [1],
              [1],
              [0]])             # (4, 1)

# ── Network dimensions ──────────────────────────────────────────────
input_dim  = 2
hidden_dim = 4
output_dim = 1

# ── Weight initialisation (small random) ────────────────────────────
W1 = np.random.randn(input_dim, hidden_dim) * 0.5   # (2, 4)
b1 = np.zeros((1, hidden_dim))                       # (1, 4)
W2 = np.random.randn(hidden_dim, output_dim) * 0.5   # (4, 1)
b2 = np.zeros((1, output_dim))                       # (1, 1)

# ── Hyperparameters ─────────────────────────────────────────────────
lr     = 2.0       # learning rate (high because dataset is tiny)
epochs = 10_000

# ── Training loop ───────────────────────────────────────────────────
for epoch in range(epochs):

    # ---------- Forward pass ----------
    z1 = X @ W1 + b1          # (4, 4)
    a1 = sigmoid(z1)          # (4, 4)  hidden activations

    z2 = a1 @ W2 + b2         # (4, 1)
    a2 = sigmoid(z2)          # (4, 1)  output predictions

    # ---------- Loss (MSE) ----------
    loss = np.mean((y - a2) ** 2)

    # ---------- Backward pass ----------
    m = X.shape[0]             # number of samples

    # Output layer gradients
    dL_da2 = -(2 / m) * (y - a2)                   # (4, 1)
    da2_dz2 = sigmoid_deriv(a2)                     # (4, 1)
    dz2 = dL_da2 * da2_dz2                          # (4, 1)  δ₂

    dW2 = a1.T @ dz2                                # (4, 1)
    db2 = np.sum(dz2, axis=0, keepdims=True)        # (1, 1)

    # Hidden layer gradients
    da1 = dz2 @ W2.T                                # (4, 4)
    dz1 = da1 * sigmoid_deriv(a1)                   # (4, 4)  δ₁

    dW1 = X.T @ dz1                                 # (2, 4)
    db1 = np.sum(dz1, axis=0, keepdims=True)        # (1, 4)

    # ---------- Parameter update (gradient descent) ----------
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # ---------- Logging ----------
    if epoch % 2000 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:>5d}  |  Loss: {loss:.6f}")

# ── Final predictions ───────────────────────────────────────────────
print("\n--- Predictions after training ---")
for i in range(len(X)):
    print(f"Input: {X[i]}  |  Predicted: {a2[i][0]:.4f}  |  Target: {y[i][0]}")

# ====================================================================
# Modern Library Implementation (PyTorch)
# ====================================================================
print("\n\n" + "="*50)
print("PyTorch Implementation")
print("="*50)

import torch
import torch.nn as nn
import torch.optim as optim

# ── XOR dataset (PyTorch tensors) ───────────────────────────────────
X_pt = torch.tensor([[0.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 0.0],
                     [1.0, 1.0]])

y_pt = torch.tensor([[0.0],
                     [1.0],
                     [1.0],
                     [0.0]])

# ── Model Definition ────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# ── Loss and Optimizer ──────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=2.0)

# ── Training loop ───────────────────────────────────────────────────
epochs_pt = 10_000

for epoch in range(epochs_pt):
    # Forward pass
    outputs = model(X_pt)
    loss = criterion(outputs, y_pt)
    
    # Backward pass and optimize
    optimizer.zero_grad()  # zero the gradient buffers
    loss.backward()        # backprop
    optimizer.step()       # update weights
    
    if epoch % 2000 == 0 or epoch == epochs_pt - 1:
        print(f"Epoch {epoch:>5d}  |  Loss: {loss.item():.6f}")

# ── Final predictions (PyTorch) ─────────────────────────────────────
print("\n--- Predictions after training (PyTorch) ---")
with torch.no_grad():
    predictions = model(X_pt)
    for i in range(len(X_pt)):
        print(f"Input: {X_pt[i].numpy()}  |  Predicted: {predictions[i].item():.4f}  |  Target: {y_pt[i].item()}")
