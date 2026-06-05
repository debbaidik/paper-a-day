"""
Day 2 — AlexNet: First-Layer Filter Visualization
Paper: Krizhevsky, Sutskever & Hinton (2012)

What this script does:
  1. Loads pretrained AlexNet (ImageNet) and a random-init copy
  2. Visualizes all 64 first-layer filters side by side
  3. Computes pairwise cosine similarity and Sobel gradient magnitude
  4. Plots dominant gradient orientation as a polar histogram

Run:
    pip install torch torchvision scipy scikit-learn matplotlib
    python day2_alexnet_filters.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import sobel
from sklearn.preprocessing import normalize
import torch
import torchvision.models as models


# ── 1. Load models ────────────────────────────────────────────────────────────

print("Loading pretrained AlexNet...")
alexnet_pretrained = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
alexnet_pretrained.eval()

print("Creating random-init AlexNet...")
torch.manual_seed(42)
alexnet_random = models.alexnet(weights=None)
alexnet_random.eval()

# First conv layer weights: (64, 3, 11, 11)
# 64 filters, 3 RGB channels, 11×11 spatial kernel
filters_pretrained = alexnet_pretrained.features[0].weight.detach().cpu().numpy()
filters_random     = alexnet_random.features[0].weight.detach().cpu().numpy()

print(f"Filter tensor shape: {filters_pretrained.shape}")
print(f"  -> 64 filters, 3 RGB channels, 11x11 pixels each\n")


# ── 2. Normalize filters for display ─────────────────────────────────────────

def normalize_filter(f):
    """(3, H, W) -> (H, W, 3) float32 in [0, 1] for imshow."""
    f = f.transpose(1, 2, 0)
    f = f - f.min()
    f = f / (f.max() + 1e-8)
    return f.astype(np.float32)

pretrained_vis = [normalize_filter(filters_pretrained[i]) for i in range(64)]
random_vis     = [normalize_filter(filters_random[i])     for i in range(64)]


# ── 3. Plot all 64 filters, pretrained vs random ──────────────────────────────

print("Plotting all 64 filters...")
fig, axes = plt.subplots(2, 64, figsize=(64 * 0.55, 3.0))
fig.patch.set_facecolor('#0f0f0f')

for row_idx, (label, filters) in enumerate(
    [('Pretrained', pretrained_vis), ('Random Init', random_vis)]
):
    for col_idx in range(64):
        ax = axes[row_idx, col_idx]
        ax.imshow(filters[col_idx], interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[row_idx, 0].set_ylabel(label, color='white', fontsize=9,
                                 rotation=90, labelpad=6, va='center')

fig.suptitle('AlexNet Layer 1 — All 64 Filters: Pretrained vs Random Init',
             color='white', fontsize=11, y=1.01)
plt.tight_layout(pad=0.2)
plt.savefig('alexnet_all_filters.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: alexnet_all_filters.png")


# ── 4. Focused view — first 16 filters ───────────────────────────────────────

print("Plotting focused 4×4 view (first 16 filters)...")
SHOW = 16

fig = plt.figure(figsize=(12, 5.5))
fig.patch.set_facecolor('#0f0f0f')
gs_outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)

for panel, (title, filters) in enumerate(
    [('Pretrained (ImageNet)', pretrained_vis), ('Random Init', random_vis)]
):
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        4, 4, subplot_spec=gs_outer[panel], hspace=0.06, wspace=0.06
    )
    for i in range(SHOW):
        ax = fig.add_subplot(gs_inner[i // 4, i % 4])
        ax.imshow(filters[i], interpolation='nearest')
        ax.axis('off')

    # Title above each panel
    title_ax = fig.add_subplot(gs_outer[panel])
    title_ax.set_title(title, color='white', fontsize=12, pad=10, fontweight='bold')
    title_ax.axis('off')

fig.suptitle('AlexNet First-Layer Filters — First 16 of 64',
             color='#aaaaaa', fontsize=10, y=1.01)
plt.savefig('alexnet_filters_focused.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: alexnet_filters_focused.png")


# ── 5. Quantitative metrics ───────────────────────────────────────────────────

def pairwise_cosine_sim(filters):
    """Mean pairwise cosine similarity over all filter pairs."""
    flat = filters.reshape(len(filters), -1).astype(np.float32)
    flat_norm = normalize(flat, norm='l2')
    sim_matrix = flat_norm @ flat_norm.T
    idx = np.triu_indices(len(filters), k=1)
    return sim_matrix[idx]

def mean_gradient_magnitude(filters):
    """Mean Sobel gradient magnitude across all filters and channels."""
    mags = []
    for f in filters:          # f: (3, 11, 11)
        for c in range(3):
            gx = sobel(f[c], axis=0)
            gy = sobel(f[c], axis=1)
            mags.append(np.sqrt(gx**2 + gy**2).mean())
    return np.array(mags)

print("\nComputing filter statistics...")
cos_pre  = pairwise_cosine_sim(filters_pretrained)
cos_rnd  = pairwise_cosine_sim(filters_random)
grad_pre = mean_gradient_magnitude(filters_pretrained)
grad_rnd = mean_gradient_magnitude(filters_random)

print("=== Pairwise Cosine Similarity (lower = more diverse) ===")
print(f"  Pretrained:  mean={cos_pre.mean():.4f}  std={cos_pre.std():.4f}")
print(f"  Random:      mean={cos_rnd.mean():.4f}  std={cos_rnd.std():.4f}")
print()
print("=== Mean Gradient Magnitude (higher = more structured) ===")
print(f"  Pretrained:  {grad_pre.mean():.4f}")
print(f"  Random:      {grad_rnd.mean():.4f}")

# Plot distributions
COLORS = {'pre': '#4fc3f7', 'rnd': '#ef9a9a'}

print("\nPlotting filter statistics...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor('#0f0f0f')

ax = axes[0]
ax.set_facecolor('#1a1a1a')
ax.hist(cos_pre, bins=50, color=COLORS['pre'], alpha=0.8, label='Pretrained', density=True)
ax.hist(cos_rnd, bins=50, color=COLORS['rnd'], alpha=0.8, label='Random',     density=True)
ax.axvline(cos_pre.mean(), color=COLORS['pre'], linestyle='--', lw=1.5,
           label=f"Pre mean={cos_pre.mean():.3f}")
ax.axvline(cos_rnd.mean(), color=COLORS['rnd'], linestyle='--', lw=1.5,
           label=f"Rnd mean={cos_rnd.mean():.3f}")
ax.set_title('Pairwise Cosine Similarity\n(lower = more diverse)', color='white', fontsize=10)
ax.set_xlabel('Cosine similarity', color='#aaaaaa')
ax.set_ylabel('Density', color='#aaaaaa')
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values(): spine.set_edgecolor('#333333')
ax.legend(fontsize=8, facecolor='#2a2a2a', labelcolor='white')

ax = axes[1]
ax.set_facecolor('#1a1a1a')
ax.hist(grad_pre, bins=30, color=COLORS['pre'], alpha=0.8, label='Pretrained', density=True)
ax.hist(grad_rnd, bins=30, color=COLORS['rnd'], alpha=0.8, label='Random',     density=True)
ax.axvline(grad_pre.mean(), color=COLORS['pre'], linestyle='--', lw=1.5,
           label=f"Pre mean={grad_pre.mean():.3f}")
ax.axvline(grad_rnd.mean(), color=COLORS['rnd'], linestyle='--', lw=1.5,
           label=f"Rnd mean={grad_rnd.mean():.3f}")
ax.set_title('Filter Gradient Magnitude\n(higher = more structured)', color='white', fontsize=10)
ax.set_xlabel('Mean Sobel magnitude', color='#aaaaaa')
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values(): spine.set_edgecolor('#333333')
ax.legend(fontsize=8, facecolor='#2a2a2a', labelcolor='white')

fig.suptitle('AlexNet Filter Statistics: Pretrained vs Random', color='white', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('alexnet_filter_stats.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: alexnet_filter_stats.png")


# ── 6. Orientation polar histogram ───────────────────────────────────────────

def dominant_orientations(f):
    """Returns dominant gradient angles (0–180°) for pixels above mean magnitude."""
    gray = f.mean(axis=0)           # average over RGB -> (11, 11)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    mask = mag > mag.mean()
    return np.degrees(np.arctan2(gy[mask], gx[mask])) % 180

print("Plotting orientation histograms...")
angles_pre = np.concatenate([dominant_orientations(filters_pretrained[i]) for i in range(64)])
angles_rnd = np.concatenate([dominant_orientations(filters_random[i])     for i in range(64)])

fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw=dict(projection='polar'))
fig.patch.set_facecolor('#0f0f0f')

bins = np.linspace(0, np.pi, 19)   # 18 bins over 180°

for ax, angles, title, color in zip(
    axes,
    [angles_pre, angles_rnd],
    ['Pretrained', 'Random Init'],
    [COLORS['pre'], COLORS['rnd']]
):
    counts, _ = np.histogram(np.radians(angles), bins=bins)
    bin_centers  = (bins[:-1] + bins[1:]) / 2
    centers_full = np.concatenate([bin_centers, bin_centers + np.pi])
    counts_full  = np.concatenate([counts, counts])
    ax.bar(centers_full, counts_full, width=np.pi / 18,
           color=color, alpha=0.85, edgecolor='none')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#888888', labelsize=7)
    ax.set_title(title, color='white', fontsize=11, pad=14)
    ax.spines['polar'].set_edgecolor('#333333')

fig.suptitle(
    'Dominant Gradient Orientation in Filters\n'
    '(pretrained = structured peaks, random = uniform)',
    color='white', fontsize=11, y=1.05
)
plt.tight_layout()
plt.savefig('alexnet_orientation.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: alexnet_orientation.png")


# ── Summary ───────────────────────────────────────────────────────────────────

print("""
Done. Four files saved:
  alexnet_all_filters.png    — all 64 filters, pretrained vs random (dense grid)
  alexnet_filters_focused.png — first 16 filters, larger view
  alexnet_filter_stats.png   — cosine similarity + gradient magnitude distributions
  alexnet_orientation.png    — polar histogram of filter orientations

Key observations:
  - Pretrained filters look like Gabor edges + color blobs; random = noise
  - Pretrained: lower pairwise cosine sim  -> diverse, each filter does something different
  - Pretrained: higher Sobel magnitude     -> sharp, structured spatial patterns
  - Pretrained orientation polar plot      -> peaks at specific angles (like V1 simple cells)
  - Random orientation polar plot          -> roughly uniform (no preferred direction)
""")