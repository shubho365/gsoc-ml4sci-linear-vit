# Multi-Architecture Vision Transformer Benchmark for Particle Collision Images

**GSoC ML4SCI Project** — Research-Grade Experimental Framework

## Overview

This repository provides a **comprehensive research pipeline** for particle collision detector image analysis using Vision Transformers. The pipeline covers:

1. **Self-supervised pretraining** — masked image modeling on unlabeled detector data
2. **Multi-task fine-tuning** — particle classification + mass regression with uncertainty weighting
3. **Pretrained vs. scratch comparison** — demonstrating the benefit of self-supervised pretraining
4. **Four-architecture benchmark** — StandardViT, LinearAttentionViT, L2ViT, XCiTViT

## Experimental Pipeline

```
Dataset_Specific_Unlabelled.h5
         │
         ▼
Step 1 ─ Self-Supervised Pretraining (MAE / MAE v2 / SimMIM)
         │  pretrained encoder weights saved
         ▼
Step 2 ─ Fine-tune Pretrained XCiTViT
         │  uncertainty-weighted multi-task loss (Kendall et al.)
         ▼
Step 3 ─ Train XCiTViT from Scratch
         │  same architecture, same loss, no pretrained weights
         ▼
Step 4 ─ Train Standard ViT, Linear Attention ViT, and L2ViT (standard CE+MSE loss)
         ▼
Final ─  5-Model Benchmark Comparison Table
```

## Architecture Details

### Standard Vision Transformer (ViT)

| Component | Description |
|-----------|-------------|
| **PatchEmbed** | Conv2d with stride=patch_size producing N = (64/8)² = 64 tokens |
| **Positional Embedding** | Learnable (1, N, D) embedding |
| **Attention** | Standard multi-head softmax: `Softmax(QKᵀ/√d)·V` |
| **Blocks** | Pre-norm: `x = x + Attn(LN(x)); x = x + FFN(LN(x))` |
| **Stochastic Depth** | Linear decay 0→0.1 over DEPTH=10 blocks |
| **Complexity** | O(N²·d) — quadratic in number of tokens |

### Linear Attention Vision Transformer (ReLU kernel)

| Component | Description |
|-----------|-------------|
| **LinearAttention** | ReLU kernel feature maps: φ(Q)=ReLU(Q), φ(K)=ReLU(K) |
| **Formula** | φ(Q)(φ(K)ᵀV) / (φ(Q)(φ(K)ᵀ1)) |
| **Blocks** | Pre-norm: `x = x + Attn(LN(x)); x = x + FFN(LN(x))` |
| **Complexity** | O(N·d²) — linear in number of tokens |

### L2ViT — Linear Global Attention + Local Window Attention

| Component | Description |
|-----------|-------------|
| **LinearGlobalAttention (LGA)** | ReLU kernel linear attention for global context |
| **LocalWindowAttention (LWA)** | Standard softmax attention within non-overlapping windows |
| **LocalConcentrationModule (LCM)** | Depth-wise convolutions to refocus dispersive linear attention |
| **ConditionalPositionalEncoding** | Depth-wise 3×3 convolution for position encoding |
| **Alternating Pattern** | LWA → LGA → LWA → LGA → ... |
| **Complexity** | O(N·d²) global + O(w²·d) local per layer |

### XCiT Vision Transformer (Cross-Covariance Attention)

| Component | Description |
|-----------|-------------|
| **CrossCovarianceAttention** | L2-normalise Q,K along token dim → d×d attention matrix, learnable temperature |
| **LocalPatchInteraction (LPI)** | Two depth-wise 3×3 convs with BatchNorm + GELU |
| **LayerScale** | CaiT-style per-channel scaling (init=1e-4) after each residual |
| **XCiT Block** | `x = x + LayerScale(XCA(LN(x))); x = x + LayerScale(LPI(LN(x))); x = x + LayerScale(FFN(LN(x)))` |
| **Complexity** | O(N·d²) — linear in number of tokens |

## Physics-Inspired Preprocessing

| Step | Transform | Purpose |
|------|-----------|---------|
| 1 | `x = log1p(x.clamp(min=0))` | Log energy compression |
| 2 | `x[x < 1e-3] = 0` | Detector noise suppression |
| 3 | `x = x / (x.sum() + 1e-8)` | Per-event energy normalization |
| 4 | `x = (x - mean) / (std + 1e-6)` | Standardization |
| 5 | Resize to (img_size, img_size) | Spatial normalization |

## Data Augmentation

Physics-aware augmentations applied during training:

- **Rotation symmetry**: `torch.rot90(x, k, dims=[-2,-1])`
- **Horizontal / vertical flips**
- **Gaussian detector noise**: `x + randn_like(x) * 0.01`
- **Energy scaling**: `x * Uniform(0.9, 1.1)`
- **Dead pixel simulation**: `x * (rand_like(x) > 0.02)`

## Self-Supervised Pretraining Approaches

Three masked image modeling methods are implemented:

### 1. MAE (Masked Autoencoder)
*He et al., CVPR 2022*

- Mask random patches (mask_ratio = 0.50)
- Encode **visible** patches only with the XCiTViT encoder
- Decode to reconstruct masked patch pixel values via lightweight transformer decoder
- Loss: MSE on masked pixels

### 2. MAE v2 (Feature Distillation)
*2023–2024 improvements*

- Uses EMA (Exponential Moving Average) teacher encoder as target
- Student encoder predicts teacher's feature representations (not raw pixels)
- More stable training, better downstream transfer

### 3. SimMIM (Simple Masked Image Modeling)
*Xie et al., CVPR 2022*

- Mask random patches with mask tokens
- Encode the full sequence (visible + masked) through the encoder
- Simple linear projection head predicts pixel values of masked patches
- Simpler architecture than MAE, competitive performance

## Multi-Task Uncertainty-Weighted Loss

Replaces the simple `CE + λ·MSE` with learnable uncertainty weighting (Kendall et al., 2018):

```python
class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with homoscedastic uncertainty weighting.
    
    Loss = CE / exp(2·log_σ₁) + log_σ₁ + MSE / exp(2·log_σ₂) + log_σ₂
    
    log_σ₁, log_σ₂ are learned parameters — the model automatically
    balances classification vs. regression importance during training.
    """
    def __init__(self):
        super().__init__()
        self.log_sigma1 = nn.Parameter(torch.zeros(1))  # classification
        self.log_sigma2 = nn.Parameter(torch.zeros(1))  # regression
```

**Reference:** Kendall, Gal & Cipolla, *"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"*, CVPR 2018.

## Configuration

All models use the same hyperparameters:

```python
IMG_SIZE = 64       # Input resolution
PATCH_SIZE = 8      # Tokens: (64/8)² = 64 per image
IN_CHANS = 8        # 8-channel detector images
EMBED_DIM = 256     # Model width
DEPTH = 10          # Transformer depth
NUM_HEADS = 8       # Attention heads
MLP_RATIO = 4.0     # FFN expansion ratio
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 35
LR = 3e-4           # AdamW learning rate
WEIGHT_DECAY = 1e-4
TRAIN_FRAC = 0.80   # 80/20 split
```

## Training Pipeline

All finetuning models are trained with:
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR with linear warmup
- **Mixed Precision**: `torch.cuda.amp` (automatic fallback on CPU/MPS)
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: patience=5, monitors val loss/MSE
- **Checkpointing**: saves best state_dict
- **Mass Normalization**: regression targets normalized to zero mean, unit std

## Evaluation Metrics

- **Classification**: Accuracy, F1 (macro), Precision, Recall, Confusion Matrix
- **Regression**: MSE, MAE, R² (coefficient of determination)
- **System**: Training time (s), inference speed (ms/sample), peak GPU memory (MB), parameter count

## Benchmark Comparison Output

The final benchmark shows all 5 training runs:

```
| Model                   | Accuracy | F1    | MSE   | MAE   | R²    | Train Time | Inference (ms) | GPU Mem (MB) | Parameters |
|-------------------------|----------|-------|-------|-------|-------|------------|----------------|--------------|------------|
| Standard ViT            |          |       |       |       |       |            |                |              |            |
| Linear Attention ViT    |          |       |       |       |       |            |                |              |            |
| L2ViT                   |          |       |       |       |       |            |                |              |            |
| XCiT ViT (pretrained)   |          |       |       |       |       |            |                |              |            |
| XCiT ViT (scratch)      |          |       |       |       |       |            |                |              |            |
```

## Model Saving

Trained weights are saved to the `models/` directory:

```
models/
    vit.pt              # Standard ViT
    linear_vit.pt       # Linear Attention ViT
    l2vit.pt            # L2ViT
    xcit.pt             # XCiT ViT (pretrained)
    pretrained_encoder.pt  # SimMIM pretrained encoder
```

With visualizations: overlaid training curves, mass scatter plots per model, confusion matrices, and bar charts comparing key metrics across all 5 models.

## Datasets

Place the following HDF5 files in a `data/` directory:

- `Dataset_Specific_Unlabelled.h5` — unlabeled detector images for pretraining
- `Dataset_Specific_labelled_full_only_for_2i.h5` — labeled images with mass and class targets

The notebook includes **synthetic fallback data** for demonstration if the files are not available, so it runs end-to-end without any data files.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── notebooks/
│   ├── linear_attention_vit.ipynb     # 16-section research notebook
│   └── nextgen_vit_functional.py      # L2ViT functional implementation
├── models/                            # Saved model weights (.pt files)
├── images/                            # Architecture diagrams
├── papers/                            # Reference research papers
├── requirements.txt                   # Python dependencies
├── README.md
└── .gitignore
```

## Notebook Sections

| Section | Content |
|---------|---------|
| 1. Configuration | All hyperparameters, device detection, seeds |
| 2. Dataset Loading | LazyHDF5Dataset (labeled + unlabeled), inspect_hdf5(), synthetic fallback |
| 3. Preprocessing | Log-compress, noise suppression, energy normalization, standardization, augmentations |
| 4. Model Architectures | StandardViT, LinearAttentionViT (ReLU kernel), L2ViT (LGA+LWA), XCiTViT (Cross-Covariance) |
| 5. Pretraining Models | MAEPretrainer, MAEv2Pretrainer, SimMIMPretrainer |
| 6. Training Utilities | train_epoch (AMP), CosineWarmupScheduler, UncertaintyWeightedLoss, run_experiment(), run_experiment_uw(), EarlyStopping |
| 7. Evaluation Metrics | Accuracy, F1, Precision, Recall, MSE, MAE, R² |
| 8. Visualization Tools | Loss curves, scatter, confusion matrix, attention maps, comparison plots |
| 9. Self-Supervised Pretrain | Pretraining on unlabeled data (SimMIM), save encoder |
| 10. Fine-tune XCiTViT | Fine-tune pretrained encoder with uncertainty-weighted loss |
| 11. Scratch XCiTViT | Train XCiTViT from scratch for comparison |
| 12. Standard ViT | Train Standard ViT with standard CE+MSE loss |
| 13. Linear Attention ViT | Train Linear Attention ViT with standard loss |
| 14. L2ViT | Train L2ViT with standard loss |
| 15. Benchmark Comparison | Model comparison table + model saving + all visualizations |
| 16. Final Results | Summary, best-per-metric analysis, pretraining benefit quantification |

## References

- **ViT**: Dosovitskiy et al., *"An Image is Worth 16×16 Words"*, ICLR 2021
- **XCiT**: El-Nouby et al., *"XCiT: Cross-Covariance Image Transformers"*, NeurIPS 2021
- **L2ViT**: Zheng, *"The Linear Attention Resurrection in Vision Transformer"*, arXiv:2501.16182, 2025
- **MAE**: He et al., *"Masked Autoencoders Are Scalable Vision Learners"*, CVPR 2022
- **SimMIM**: Xie et al., *"SimMIM: A Simple Framework for Masked Image Modeling"*, CVPR 2022
- **LayerScale/CaiT**: Touvron et al., *"Going deeper with Image Transformers"*, ICCV 2021
- **RoPE**: Su et al., *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*, 2021
- **Uncertainty Loss**: Kendall et al., *"Multi-Task Learning Using Uncertainty to Weigh Losses"*, CVPR 2018

## Full Implementation

[`notebooks/linear_attention_vit.ipynb`](notebooks/linear_attention_vit.ipynb)
