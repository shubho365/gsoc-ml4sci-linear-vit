# Multi-Architecture Vision Transformer Benchmark for Particle Collision Images

**GSoC ML4SCI Project** — Research-Grade Experimental Framework

## Overview

This repository provides a **comprehensive research pipeline** for particle collision detector image analysis using Vision Transformers. The pipeline covers:

1. **Self-supervised pretraining** — masked image modeling on unlabeled detector data
2. **Multi-task fine-tuning** — particle classification + mass regression with uncertainty weighting
3. **Pretrained vs. scratch comparison** — demonstrating the benefit of self-supervised pretraining
4. **Five-model benchmark** — comprehensive comparison across architectures and training strategies

## Experimental Pipeline

```
Dataset_Specific_Unlabelled.h5
         │
         ▼
Step 1 ─ Self-Supervised Pretraining (MAE / MAE v2 / SimMIM)
         │  pretrained encoder weights saved
         ▼
Step 2 ─ Fine-tune Pretrained LinearAttentionViT
         │  uncertainty-weighted multi-task loss (Kendall et al.)
         ▼
Step 3 ─ Train LinearAttentionViT from Scratch
         │  same architecture, same loss, no pretrained weights
         ▼
Step 4 ─ Train Standard ViT & Hybrid CNN+ViT (standard CE+MSE loss)
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

### Linear Attention Vision Transformer (XCA-based + Modern Techniques)

| Component | Description |
|-----------|-------------|
| **CrossCovarianceAttention** | L2-normalise Q,K along token dim → d×d attention matrix, learnable temperature |
| **LocalPatchInteraction (LPI)** | Two depth-wise 3×3 convs with BatchNorm + GELU |
| **RoPE** | Rotary Positional Embeddings applied to Q and K in XCA |
| **Relative Position Bias** | Learnable bias table added to attention logits |
| **LayerScale** | CaiT-style per-channel scaling (init=1e-4) after each residual |
| **Token Pruning** | Score-based token removal for efficient inference |
| **XCiT Block** | `x = x + LayerScale(XCA(LN(x))); x = x + LayerScale(LPI(LN(x))); x = x + LayerScale(FFN(LN(x)))` |
| **Complexity** | O(N·d²) — linear in number of tokens |

### Hybrid CNN + Vision Transformer

| Component | Description |
|-----------|-------------|
| **CNN Block 1** | IN_CHANS → 64 channels, MaxPool2d(2): 64→32 spatial |
| **CNN Block 2** | 64 → 128 channels, MaxPool2d(2): 32→16 spatial |
| **CNN Block 3** | 128 → EMBED_DIM channels, MaxPool2d(2): 16→8 spatial |
| **Transformer** | DEPTH//2 standard ViT blocks on 8×8=64 tokens |
| **Benefit** | CNN inductive biases + global transformer context |

## Self-Supervised Pretraining Approaches

Three masked image modeling methods are implemented:

### 1. MAE (Masked Autoencoder)
*He et al., CVPR 2022*

- Mask random patches (mask_ratio = 0.50)
- Encode **visible** patches only with the LinearAttentionViT encoder
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
EPOCHS = 20
LR = 3e-4           # AdamW learning rate
WEIGHT_DECAY = 1e-4
TRAIN_FRAC = 0.80   # 80/20 split
```

## Training Pipeline

All finetuning models are trained with:
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Mixed Precision**: `torch.cuda.amp` (automatic fallback on CPU/MPS)
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: patience=5, monitors val MSE
- **Checkpointing**: saves best state_dict by val MSE

## Evaluation Metrics

- **Classification**: Accuracy, F1 (macro), Precision, Recall, Confusion Matrix
- **Regression**: MSE, MAE, R² (coefficient of determination)
- **System**: Training time (s), peak GPU memory (MB), parameter count

## Benchmark Comparison Output

The final benchmark shows all 5 training runs:

```
| Model                   | Accuracy | F1    | MSE   | MAE   | R²    | Train Time (s) | Parameters |
|-------------------------|----------|-------|-------|-------|-------|----------------|------------|
| Standard ViT            |          |       |       |       |       |                |            |
| Linear Attention ViT    |          |       |       |       |       |                |            |
| Hybrid CNN+ViT          |          |       |       |       |       |                |            |
| Linear ViT (pretrained) |          |       |       |       |       |                |            |
| Linear ViT (scratch)    |          |       |       |       |       |                |            |
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
│   ├── linear_attention_vit.ipynb     # 15-section research notebook
│   └── nextgen_vit_functional.py      # L2ViT functional implementation
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
| 3. Preprocessing | Log-compress, resize to 64×64, 8-channel, augmentations |
| 4. Model Architectures | StandardViT, LinearAttentionViT (with RoPE, LayerScale, Token Pruning, Rel. Pos. Bias), HybridCNNViT |
| 5. Pretraining Models | MAEPretrainer, MAEv2Pretrainer, SimMIMPretrainer |
| 6. Training Utilities | train_epoch (AMP), UncertaintyWeightedLoss, run_experiment(), run_experiment_uw(), EarlyStopping |
| 7. Evaluation Metrics | Accuracy, F1, Precision, Recall, MSE, MAE, R² |
| 8. Visualization Tools | Loss curves, scatter, confusion matrix, attention maps, comparison plots |
| 9. Experiment 1: Pretrain | Self-supervised pretraining on unlabeled data (SimMIM), save encoder |
| 10. Experiment 2: Finetune | Fine-tune pretrained encoder with uncertainty-weighted loss |
| 11. Experiment 3: Scratch | Train LinearAttentionViT from scratch (standard + uncertainty-weighted) |
| 12. Experiment 4: Std ViT | Train Standard ViT with standard loss |
| 13. Experiment 5: Hybrid | Train Hybrid CNN+ViT with standard loss |
| 14. Benchmark Comparison | 5-model comparison table + all visualizations |
| 15. Final Results | Summary, best-per-metric analysis, pretraining benefit quantification |

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
