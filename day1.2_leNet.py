"""
LeNet-5 on MNIST (PyTorch)
==========================
Paper : "Gradient-Based Learning Applied to Document Recognition"
Authors: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner (1998)

Architecture (original):
    Input (1×32×32)
    → Conv1  (6  filters, 5×5) → Tanh → AvgPool (2×2)
    → Conv2  (16 filters, 5×5) → Tanh → AvgPool (2×2)
    → Flatten
    → FC1 (400 → 120)  → Tanh
    → FC2 (120 → 84)   → Tanh
    → FC3 (84  → 10)

Note: MNIST images are 1×28×28, so we pad them to 1×32×32 to match
the original LeNet-5 input size.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ====================================================================
# 1. Hyperparameters
# ====================================================================
BATCH_SIZE    = 64
LEARNING_RATE = 0.05
EPOCHS        = 5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================
# 2. Dataset — MNIST with padding to 32×32
# ====================================================================
transform = transforms.Compose([
    transforms.Pad(2),                        # 28×28 → 32×32
    transforms.ToTensor(),                    # [0,255] → [0.0,1.0]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std
])

# Override MNIST URLs to use a reliable mirror (default host can be flaky)
datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ====================================================================
# 3. LeNet-5 Model
# ====================================================================
class LeNet5(nn.Module):
    """
    Faithful reproduction of the LeNet-5 architecture.
    Uses Tanh activations and average pooling like the original paper.
    """

    def __init__(self):
        super().__init__()

        # --- Feature extractor (conv layers) ---
        self.features = nn.Sequential(
            # Layer C1: 6 feature maps, 5×5 kernel  →  output: 6×28×28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh(),
            # Layer S2: average pooling 2×2          →  output: 6×14×14
            nn.AvgPool2d(kernel_size=2, stride=2),

            # Layer C3: 16 feature maps, 5×5 kernel →  output: 16×10×10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            # Layer S4: average pooling 2×2          →  output: 16×5×5
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # --- Classifier (fully connected layers) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 16×5×5 = 400
            nn.Linear(16 * 5 * 5, 120),         # Layer C5
            nn.Tanh(),
            nn.Linear(120, 84),                 # Layer F6
            nn.Tanh(),
            nn.Linear(84, 10),                  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ====================================================================
# 4. Training Loop
# ====================================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 200 == 0:
            avg = running_loss / 200
            print(f"  Epoch [{epoch+1}/{EPOCHS}]  "
                  f"Batch [{batch_idx+1}/{len(loader)}]  "
                  f"Loss: {avg:.4f}")
            running_loss = 0.0

# ====================================================================
# 5. Evaluation Loop
# ====================================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs  = model(images)
            _, preds = torch.max(outputs, dim=1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy

# ====================================================================
# 6. Main
# ====================================================================
if __name__ == "__main__":
    model     = LeNet5().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    print(f"Device : {DEVICE}")
    print(f"Model  :\n{model}\n")

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        acc = evaluate(model, test_loader)
        print(f"  -> Test Accuracy after epoch {epoch+1}: {acc:.2f}%\n")

    print("Training complete.")
