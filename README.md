# Linear Attention Vision Transformers for End-to-End Mass Regression and Classification

**GSoC ML4SCI Project** — PyTorch Implementation

## Overview

This repository implements a **Linear Attention Vision Transformer** trained on particle collision detector images to perform:

1. **Particle mass regression** — predict the mass of detected particles
2. **Particle type classification** — classify the type of detected particle

The model is based on **Cross-Covariance Attention (XCA)** from the XCiT architecture, which achieves linear complexity O(N·d²) in the number of tokens instead of the O(N²·d) of standard self-attention.

## Architecture

The model implements **L2ViT** (arXiv:2501.16182) — *"The Linear Attention Resurrection in Vision Transformer"* — adapted for CMS 125×125 jet images.

| Component | Description |
|-----------|-------------|
| **Convolutional Stem** | Three stride-2 Conv2D layers reducing 125×125 → 16×16 tokens |
| **Conditional Positional Encoding (CPE)** | Depth-wise 3×3 conv residual for resolution-agnostic position |
| **ReLU-based Linear Attention (LA)** | φ(Q)(φ(K)ᵀV)·s / clamp(φ(Q)·Σφ(K), 1e2), φ=ReLU — O(N·C²) |
| **Local Concentration Module (LCM)** | DWConv₁(7×7)→GELU→BN→DWConv₂(7×7) applied as residual Y=LCM(LN(X))+X |
| **Linear Global Attention (LGA) block** | CPE → ReLU-LA + LCM residual → FFN |
| **Local Window Attention (LWA) block** | CPE → window self-attention (4×4) → FFN |
| **Alternating LGA + LWA** | LWA first captures local features, LGA builds global context |
| **Regression Head** | MLP predicting normalized particle mass |
| **Classification Head** | MLP predicting particle class logits (quark vs. gluon) |

## Training Pipeline

1. **Pretrain** the encoder on the unlabeled dataset using masked patch reconstruction (MAE-style)
2. **Finetune** the full model on the labeled dataset (80% train / 20% validation) with a two-phase warmup strategy
3. **Train from scratch** an identical model as a baseline for comparison

Both finetuned and scratch models use **best-model checkpointing** (based on validation MSE) to prevent overfitting.

## Evaluation Metrics

- **Regression**: MSE, MAE, R² (coefficient of determination)
- **Classification**: Accuracy, F1 score, precision, recall, per-class accuracy, confusion matrix
- **Visualization**: Loss curves, MSE/accuracy vs epoch, predicted vs true mass scatter, error distribution histogram, learning rate schedules, convergence comparison

## Datasets

Place the following HDF5 files in a `data/` directory:

- `Dataset_Specific_Unlabelled.h5` — unlabeled detector images for pretraining
- `Dataset_Specific_labelled_full_only_for_2i.h5` — labeled images with mass and class targets

The notebook includes synthetic fallback data for demonstration if the files are not available.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── notebooks/
│   └── linear_attention_vit.ipynb   # Full implementation notebook
├── images/                          # Architecture diagrams from reference papers
├── papers/                          # Reference research papers
├── requirements.txt                 # Python dependencies
├── README.md
└── .gitignore
```

## References

- **L2ViT**: Zheng, "The Linear Attention Resurrection in Vision Transformer", arXiv:2501.16182, 2025
- **XCiT**: El-Nouby et al., "XCiT: Cross-Covariance Image Transformers", NeurIPS 2021
- **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021

## Notebook

Full implementation: [`notebooks/linear_attention_vit.ipynb`](notebooks/linear_attention_vit.ipynb)
