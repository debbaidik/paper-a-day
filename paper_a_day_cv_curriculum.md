# Paper-a-Day Curriculum: Modern Computer Vision & Representation Learning
*Experiments reframed for Colab (free tier / T4) — focus on intuition-building, not training runs*

---

## Week 1 — CNN Foundations

| Day | Paper | Core Idea | Colab Experiment |
|-----|-------|-----------|-----------------|
| 1 | Rumelhart et al. (1986) — Backpropagation + LeNet-5 (1998) | Gradient-based learning and CNNs | Implement backprop by hand in NumPy on a 2-layer net; verify gradients with finite differences |
| 2 | AlexNet (2012) | Deep CNN breakthrough | Load a pretrained AlexNet; visualize first-layer filters; compare to random-init filters |
| 3 | Visualizing and Understanding CNNs (2014) | Feature visualization | Apply gradient-based saliency maps to a pretrained ResNet on ImageNet samples |
| 4 | VGG (2014) | Depth matters | Extract intermediate feature maps from VGG-16 at different depths; compare spatial resolution and abstraction level |
| 5 | Batch Normalization (2015) | Optimization stability | Plot activation distributions before/after BN in a small net; observe covariate shift disappearing |
| 6 | ResNet (2015) | Residual learning | Visualize gradient magnitudes across layers with and without skip connections using hooks |
| 7 | ConvNeXt (2022) | Modern CNN redesign | Compare CKA similarity matrices between ResNet-50 and ConvNeXt-T feature spaces on 500 ImageNet samples |

---

## Week 2 — Optimization and Transformers

| Day | Paper | Core Idea | Colab Experiment |
|-----|-------|-----------|-----------------|
| 8 | Layer Normalization (2016) | Transformer-friendly normalization | Plot activation statistics across a ViT's layers with LN; compare to a CNN's BN statistics |
| 9 | AdamW (2017) | Decoupled weight decay | Train a tiny MLP on CIFAR-10 with Adam vs AdamW; plot weight norm trajectories |
| 10 | Attention Is All You Need (2017) | Self-attention | Implement scaled dot-product attention from scratch; visualize attention patterns on a toy sequence |
| 11 | Vision Transformer (ViT) (2020) | Images as patches | Load pretrained ViT-B/16; visualize positional embeddings and patch token similarity maps |
| 12 | DeiT (2021) | Data-efficient ViTs | Compare CLS token representations from ViT and DeiT via t-SNE on 1000 ImageNet samples |
| 13 | Swin Transformer (2021) | Hierarchical transformers | Visualize window-partitioned attention maps across Swin stages; compare receptive field growth |
| 14 | Hiera (2023) | Simpler hierarchical transformers | Extract and compare feature pyramid structure from Swin vs Hiera using pretrained weights |

---

## Week 3 — Representation Learning Foundations

| Day | Paper | Core Idea | Colab Experiment |
|-----|-------|-----------|-----------------|
| 15 | Do Vision Transformers See Like CNNs? (2021) | Representation similarity | Compute CKA between ResNet-50 and ViT-B/16 layer pairs on 500 images; reproduce the paper's main finding |
| 16 | Intriguing Properties of Vision Transformers (2021) | Robustness characteristics | Add patch occlusion to images; measure ViT vs ResNet top-1 drop as occluded patch count increases |
| 17 | How Do Vision Transformers Work? (2022) | Optimization geometry | Visualize loss landscape around a pretrained ViT checkpoint using filter normalization |
| 18 | Contrastive Predictive Coding (2018) | Predictive representation learning | Implement a 1D CPC toy experiment: predict future embeddings in a random walk sequence |
| 19 | DeepCluster (2018) | Clustering-based SSL | Run k-means on frozen ResNet features; visualize cluster purity vs ImageNet classes with a confusion matrix |
| 20 | MoCo (2019) | Momentum contrast | Load pretrained MoCo-v2 weights; compare linear probe accuracy to supervised ResNet on frozen features |
| 21 | SimCLR (2020) | Large-scale contrastive learning | Load pretrained SimCLR weights; visualize augmentation-invariance by plotting embedding pairs before/after augmentation |
| 22 | BYOL (2020) | Non-contrastive SSL | Load pretrained BYOL; compare feature kNN clusters to SimCLR on the same 1000 images |

---

## Week 4 — Modern Self-Supervised Learning

| Day | Paper | Core Idea | Colab Experiment |
|-----|-------|-----------|-----------------|
| 23 | SimSiam (2021) | Stop-gradient learning | Inspect cosine similarity between online and target branch outputs; visualize what happens to the similarity curve when you remove stop-grad (conceptual trace through the math) |
| 24 | VICReg (2021) | Variance-Invariance-Covariance Regularization | Load pretrained VICReg; compute and plot the variance and covariance matrices of embeddings over a mini-batch; observe the regularization effect |
| 25 | MAE (2021) | Masked image modeling | Load pretrained MAE; run inference with 75% masking on your own images; visualize reconstructions and identify failure cases |
| 26 | DINO (2021) | Self-distillation | Load DINO ViT-S/8; extract and visualize self-attention maps on 10 images; observe foreground segmentation emergence |
| 27 | iBOT (2021) | Token-level SSL | Compare DINO vs iBOT patch-token attention maps; measure how much more spatially grounded iBOT tokens are |
| 28 | CLIP (2021) | Vision-language alignment | Run zero-shot classification on 200 CIFAR-100 images; probe where CLIP fails by finding confident wrong predictions |
| 29 | SigLIP (2023) | Sigmoid contrastive learning | Compare CLIP vs SigLIP embedding similarities on the same image-text pairs; visualize the score distributions |
| 30 | DINOv2 (2023) | Universal visual features | Run DINOv2 frozen features on a domain-shifted dataset (e.g., sketch or medical images); compare linear probe vs kNN accuracy |
| 31 | I-JEPA (2023) | Predictive world models | Load pretrained I-JEPA; compare its internal representations to MAE via CKA; note where prediction-in-latent-space differs from pixel reconstruction |

---

## Coverage Map

- CNNs: LeNet, AlexNet, VGG, ResNet, ConvNeXt
- Vision Transformers: ViT, DeiT, Swin, Hiera
- Predictive SSL: CPC, I-JEPA
- Clustering SSL: DeepCluster
- Contrastive SSL: MoCo, SimCLR
- Bootstrap SSL: BYOL
- Stop-Gradient SSL: SimSiam
- Variance/Covariance SSL: VICReg
- Masked Modeling: MAE
- Self-Distillation: DINO, DINOv2
- Token SSL: iBOT
- Vision-Language: CLIP, SigLIP

---

## Colab Constraints & Philosophy

**Target hardware**: Free Colab T4 (16GB VRAM) or CPU-only for analysis experiments  
**No training from scratch**: All experiments use pretrained weights (torchvision, timm, HuggingFace)  
**Experiment philosophy**: The goal is not to reproduce benchmarks — it is to *see* what the paper claims. Each experiment is designed to give you one clear visual or numerical insight that anchors the paper's core idea in memory.

**Common tools across experiments**:
- `timm` — pretrained model zoo
- `torchvision` — standard pretrained CNNs and datasets
- `transformers` / `open_clip` — CLIP, SigLIP, MAE, iBOT
- `sklearn` — kNN, k-means, PCA, t-SNE
- `matplotlib` / `seaborn` — all visualization
- `torch.nn.functional.cosine_similarity`, `torch.linalg` — quick representation analysis

**CKA note** (Days 7, 15, 31): Use the linear CKA implementation from Kornblith et al. — ~20 lines of NumPy, runs on CPU in seconds for batches of 500 images.
