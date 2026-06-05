"""
Day 3 — Visualizing and Understanding CNNs (Zeiler & Fergus, 2014)
Paper: Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks"

What this script does:
  1. Loads a pretrained ResNet-50 (ImageNet) via torchvision
  2. Downloads sample ImageNet images from torchvision
  3. Computes vanilla gradient-based saliency maps (∂class_score / ∂input)
  4. Visualizes saliency overlays to show which pixels the network attends to

Run:
    pip install torch torchvision matplotlib requests Pillow
    python VisualizingCNN.py
"""

import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import requests


# ── 1. Load pretrained ResNet-50 ──────────────────────────────────────────────

print("Loading pretrained ResNet-50 (ImageNet)...")
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.eval()

# Grab the human-readable ImageNet class labels
imagenet_categories = models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

print(f"Model: ResNet-50")
print(f"  Parameters: {sum(p.numel() for p in resnet.parameters()):,}")
print(f"  Classes: {len(imagenet_categories)}")
print()


# ── 2. Image preprocessing ───────────────────────────────────────────────────

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Inverse normalization for display
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)



# ── 3. Sample images ─────────────────────────────────────────────────────────
# Strategy: try Wikimedia first (cached locally), fall back to CIFAR-10

CACHE_DIR = Path(__file__).parent / "data" / "sample_images"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "PaperADay/1.0 (educational; visualization experiment)"}

# Public-domain Wikimedia Commons images
SAMPLE_IMAGES = {
    "Golden Retriever": "https://upload.wikimedia.org/wikipedia/commons/b/bd/Golden_Retriever_Dukedestination.jpg",
    "Tabby Cat": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
    "Sports Car": "https://upload.wikimedia.org/wikipedia/commons/3/3e/Lamborghini_Gallardo_LP570-4_Spyder_Performante_%287936654522%29.jpg",
}


def load_sample_image(name, url, max_retries=3):
    """Download and cache a sample image. Returns PIL Image or None."""
    cache_path = CACHE_DIR / f"{name.replace(' ', '_').lower()}.jpg"

    # Use cached version if available
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    # Download with retry + exponential backoff
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(cache_path)  # cache for next time
            return img
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
    return None


def load_cifar10_fallback():
    """Fall back to CIFAR-10 test images (always downloads reliably from PyTorch)."""
    print("  Falling back to CIFAR-10 sample images...")
    cifar = datasets.CIFAR10(
        root=str(CACHE_DIR.parent), train=False, download=True,
    )
    # CIFAR-10 class names that overlap with ImageNet
    # 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
    # 5=dog, 6=frog, 7=horse, 8=ship, 9=truck
    targets = {"Dog": 5, "Cat": 3, "Automobile": 1}
    result = {}
    for label_name, class_idx in targets.items():
        # Find first image of this class
        for i, (img, lbl) in enumerate(cifar):
            if lbl == class_idx:
                # Upscale 32x32 -> 256x256 for better visualization
                img = img.resize((256, 256), Image.BICUBIC)
                result[f"{label_name} (CIFAR-10)"] = img
                break
    return result


print("Loading sample images...")
images = {}
for name, url in SAMPLE_IMAGES.items():
    img = load_sample_image(name, url)
    if img is not None:
        images[name] = img
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name}")
    time.sleep(1)  # be nice to the server

if not images:
    images = load_cifar10_fallback()
    for name in images:
        print(f"  [OK] {name}")
print()


# ── 4. Compute saliency maps ─────────────────────────────────────────────────

def compute_saliency(model, input_tensor):
    """
    Vanilla gradient saliency: take the gradient of the predicted class score
    with respect to the input image, then take the max absolute value across
    the RGB channels.

    Returns:
        saliency  — (H, W) numpy array, normalized to [0, 1]
        pred_idx  — predicted class index
        pred_prob — predicted class probability
    """
    input_tensor = input_tensor.unsqueeze(0)  # add batch dim
    input_tensor.requires_grad_(True)

    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    pred_idx = logits.argmax(dim=1).item()
    pred_prob = probs[0, pred_idx].item()

    # Backprop the predicted class score
    model.zero_grad()
    logits[0, pred_idx].backward()

    # Saliency = max |∂score/∂pixel| over RGB channels
    saliency = input_tensor.grad.data.abs().squeeze(0)  # (3, H, W)
    saliency, _ = saliency.max(dim=0)                   # (H, W)
    saliency = saliency.numpy()

    # Normalize to [0, 1]
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)

    return saliency, pred_idx, pred_prob


print("Computing saliency maps...")
results = {}
for name, pil_img in images.items():
    tensor = preprocess(pil_img)
    saliency, pred_idx, pred_prob = compute_saliency(resnet, tensor)

    # Recover the display image (de-normalized)
    display_img = inv_normalize(tensor).clamp(0, 1).permute(1, 2, 0).numpy()

    pred_label = imagenet_categories[pred_idx]
    results[name] = {
        "display": display_img,
        "saliency": saliency,
        "pred_label": pred_label,
        "pred_prob": pred_prob,
    }
    print(f"  {name} -> predicted: {pred_label} ({pred_prob:.1%})")
print()


# ── 5. Visualize ──────────────────────────────────────────────────────────────

n = len(results)
fig, axes = plt.subplots(3, n, figsize=(5 * n, 13))
fig.patch.set_facecolor("#0f0f0f")

if n == 1:
    axes = axes.reshape(-1, 1)

for col, (name, res) in enumerate(results.items()):
    # Row 0: Original image
    ax = axes[0, col]
    ax.imshow(res["display"])
    ax.set_title(
        f'{name}\nPredicted: {res["pred_label"]} ({res["pred_prob"]:.1%})',
        color="white", fontsize=10, pad=8,
    )
    ax.axis("off")

    # Row 1: Raw saliency map
    ax = axes[1, col]
    ax.imshow(res["saliency"], cmap="inferno")
    ax.set_title("Saliency Map", color="white", fontsize=10, pad=8)
    ax.axis("off")

    # Row 2: Overlay
    ax = axes[2, col]
    ax.imshow(res["display"])
    ax.imshow(res["saliency"], cmap="inferno", alpha=0.5)
    ax.set_title("Overlay", color="white", fontsize=10, pad=8)
    ax.axis("off")

# Row labels
for row, label in enumerate(["Original", "Saliency", "Overlay"]):
    axes[row, 0].set_ylabel(
        label, color="white", fontsize=11, rotation=90, labelpad=12, va="center"
    )

fig.suptitle(
    "Gradient-Based Saliency Maps — Pretrained ResNet-50\n"
    "(Inspired by Zeiler & Fergus 2014)",
    color="white", fontsize=14, y=1.01,
)
plt.tight_layout(pad=1.0)
plt.savefig(
    "resnet_saliency_maps.png", dpi=150, bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close()
print("Saved: resnet_saliency_maps.png")


# ── Summary ───────────────────────────────────────────────────────────────────

print("""
Done. Output saved:
  resnet_saliency_maps.png -- original, saliency, and overlay for each sample image

Key observations to look for:
  - Saliency concentrates on the object the network classifies (e.g., dog's face, car body)
  - Background regions have near-zero gradient -- the network "ignores" them
  - Fine-grained textures (fur, edges) light up because conv filters are edge-sensitive
  - This is the simplest form of "feature visualization" from Zeiler & Fergus
""")
