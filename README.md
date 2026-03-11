# Multi-Architecture Vision Transformer Benchmark for Particle Collision Images

**GSoC ML4SCI Project** — Research-Grade Experimental Framework

## Overview

This repository provides a **comprehensive benchmark** of three Vision Transformer architectures trained on particle collision detector images to perform:

1. **Particle mass regression** — predict the mass of detected particles
2. **Particle type classification** — classify the type of detected particle

Three architectures are compared side-by-side on the same dataset split and training configuration:

| Architecture | Attention Mechanism | Complexity | Description |
|---|---|---|---|
| **Standard ViT** | Softmax self-attention | O(N²·d) | Classic ViT with learnable positional embeddings |
| **Linear Attention ViT** | Cross-Covariance (XCA) | O(N·d²) | XCiT-style linear attention + Local Patch Interaction |
| **Hybrid CNN+ViT** | CNN stem + Softmax attn | Mixed | CNN backbone tokenizes spatial features for a transformer encoder |

## Architecture Details

### Standard Vision Transformer (ViT)

| Component | Description |
|-----------|-------------|
| **PatchEmbed** | Conv2d with stride=patch_size producing N = (64/8)² = 64 tokens |
| **Positional Embedding** | Learnable (1, N, D) embedding |
| **Attention** | Standard multi-head softmax: `Softmax(QKᵀ/√d)·V` |
| **Blocks** | Pre-norm: `x = x + Attn(LN(x)); x = x + FFN(LN(x))` |
| **Stochastic Depth** | Linear decay 0→0.1 over DEPTH=10 blocks |

### Linear Attention Vision Transformer (XCA-based)

| Component | Description |
|-----------|-------------|
| **CrossCovarianceAttention** | L2-normalise Q,K along token dim → d×d attention matrix, learnable temperature |
| **LocalPatchInteraction (LPI)** | Two depth-wise 3×3 convs with BatchNorm + GELU |
| **XCiT Block** | `x = x + XCA(LN(x)); x = x + LPI(LN(x)); x = x + FFN(LN(x))` |
| **Complexity** | O(N·d²) — linear in number of tokens |

### Hybrid CNN + Vision Transformer

| Component | Description |
|-----------|-------------|
| **CNN Block 1** | IN_CHANS → 64 channels, MaxPool2d(2): 64→32 spatial |
| **CNN Block 2** | 64 → 128 channels, MaxPool2d(2): 32→16 spatial |
| **CNN Block 3** | 128 → EMBED_DIM channels, MaxPool2d(2): 16→8 spatial |
| **Transformer** | DEPTH//2 standard ViT blocks on 8×8=64 tokens |
| **Benefit** | CNN inductive biases + global transformer context |

## Configuration

All models use the same hyperparameters:

```python
IMG_SIZE = 64       # Input resolution (upgraded from 32×32)
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

All three models are trained with:
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Loss**: CrossEntropy + λ·MSE (λ=1.0)
- **Mixed Precision**: `torch.cuda.amp` (automatic fallback on CPU/MPS)
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: patience=5, monitors val MSE
- **Checkpointing**: saves best state_dict by val MSE

## Evaluation Metrics

- **Classification**: Accuracy, F1 (macro), Precision, Recall, Confusion Matrix
- **Regression**: MSE, MAE, R² (coefficient of determination)
- **System**: Training time, peak GPU memory, parameter count

## Benchmark Comparison Output

After training, the notebook generates:

```
| Model                | Accuracy | F1    | MSE   | MAE   | R²    | Train Time (s) | Parameters |
|----------------------|----------|-------|-------|-------|-------|----------------|------------|
| Standard ViT         |          |       |       |       |       |                |            |
| Linear Attention ViT |          |       |       |       |       |                |            |
| Hybrid CNN+ViT       |          |       |       |       |       |                |            |
```

With visualizations: overlaid training curves, mass scatter plots, confusion matrices, and bar charts comparing key metrics.

## Datasets

Place the following HDF5 files in a `data/` directory:

- `Dataset_Specific_Unlabelled.h5` — unlabeled detector images (optional)
- `Dataset_Specific_labelled_full_only_for_2i.h5` — labeled images with mass and class targets

The notebook includes **synthetic fallback data** for demonstration if the files are not available, so it runs end-to-end without any data files.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── notebooks/
│   └── linear_attention_vit.ipynb   # 10-section benchmark notebook
├── images/                          # Architecture diagrams
├── papers/                          # Reference research papers
├── requirements.txt                 # Python dependencies
├── README.md
└── .gitignore
```

## Notebook Sections

| Section | Content |
|---------|---------|
| 1. Configuration | All hyperparameters, device detection, seeds |
| 2. Dataset Loading | LazyHDF5Dataset, inspect_hdf5(), synthetic fallback |
| 3. Preprocessing | Log-compress, resize to 64×64, 8-channel, augmentations |
| 4. Model Architectures | StandardViT, LinearAttentionViT, HybridCNNViT + DropPath |
| 5. Training Utilities | train_epoch (AMP), evaluate_model, EarlyStopping, checkpointing |
| 6. Evaluation Metrics | Accuracy, F1, Precision, Recall, MSE, MAE, R² |
| 7. Visualization Tools | Loss curves, scatter, confusion matrix, comparison plots |
| 8. Experiments | run_experiment() for all 3 models |
| 9. Benchmark Comparison | pandas DataFrame table + all comparison plots |
| 10. Final Results | Summary, best model per metric, analysis discussion |

## References

- **ViT**: Dosovitskiy et al., *"An Image is Worth 16×16 Words"*, ICLR 2021
- **XCiT**: El-Nouby et al., *"XCiT: Cross-Covariance Image Transformers"*, NeurIPS 2021
- **L2ViT**: Zheng, *"The Linear Attention Resurrection in Vision Transformer"*, arXiv:2501.16182, 2025
- **MAE**: He et al., *"Masked Autoencoders Are Scalable Vision Learners"*, CVPR 2022

## Full Implementation

[`notebooks/linear_attention_vit.ipynb`](notebooks/linear_attention_vit.ipynb)
