import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. Architecture & Initialization
# ==========================================
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # Feature Extractor
        self.features = nn.Sequential(
            # Layer 1: Conv -> ReLU -> LRN -> MaxPool 
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), # Index 0
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2: Conv -> ReLU -> LRN -> MaxPool 
            nn.Conv2d(96, 256, kernel_size=5, padding=2),          # Index 4
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3: Conv -> ReLU 
            nn.Conv2d(256, 384, kernel_size=3, padding=1),         # Index 8
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv -> ReLU 
            nn.Conv2d(384, 384, kernel_size=3, padding=1),         # Index 10
            nn.ReLU(inplace=True),
            
            # Layer 5: Conv -> ReLU -> MaxPool 
            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # Index 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),                          # Index 1
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),                                 # Index 4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),                          # Index 6
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights from a zero-mean Gaussian with std=0.01
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Re-initialize specific biases to 1 to provide ReLUs with positive inputs early on
        # 2nd, 4th, and 5th convolutional layers
        nn.init.constant_(self.features[4].bias, 1)
        nn.init.constant_(self.features[10].bias, 1)
        nn.init.constant_(self.features[12].bias, 1)
        
        # Fully-connected hidden layers
        nn.init.constant_(self.classifier[1].bias, 1)
        nn.init.constant_(self.classifier[4].bias, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 2. Data Augmentation & Loading
# ==========================================
def get_dataloaders(batch_size=128):
    # The paper extracts random 224x224 patches from 256x256 images and applies horizontal reflections
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Modern approximation of their PCA RGB shift
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Subtracting mean activity
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Placeholder for ImageNet directory - requires downloading the actual dataset
    train_dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(root='./data/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# ==========================================
# 3. Training Loop Setup
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = AlexNet(num_classes=1000).to(device)
    
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.01,           # Initial learning rate
        momentum=0.9,      # Momentum
        weight_decay=0.0005 # Weight decay
    )
    
    # Learning Rate Scheduler: Divide LR by 10 when validation error stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Data Loaders
    # Note: You need the ImageNet dataset downloaded to use this locally
    # train_loader, val_loader = get_dataloaders(batch_size=128)
    
    epochs = 90
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    # --- Dummy loop for structural demonstration ---
    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0.0
    #     
    #     for inputs, labels in train_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         
    #         running_loss += loss.item()
    #         
    #     # Validation phase would go here to calculate val_loss
    #     val_loss = 0.0 
    #     scheduler.step(val_loss)
    #     print(f"Epoch {epoch+1}/{epochs} completed.")

if __name__ == "__main__":
    train_model()