# Linear Attention Vision Transformers for End-to-End Mass Regression and Classification

**GSoC ML4SCI Project** — PyTorch Implementation

## Overview

This repository implements a **Linear Attention Vision Transformer** trained on particle collision detector images to perform:

1. **Particle mass regression** — predict the mass of detected particles
2. **Particle type classification** — classify the type of detected particle

The model is based on **Cross-Covariance Attention (XCA)** from the XCiT architecture, which achieves linear complexity O(N·d²) in the number of tokens instead of the O(N²·d) of standard self-attention.

## Architecture

| Component | Description |
|-----------|-------------|
| **Patch Embedding** | Conv2D projection of non-overlapping image patches |
| **Cross-Covariance Attention (XCA)** | Linear-complexity attention over feature channels |
| **Local Patch Interaction (LPI)** | Depth-wise convolutions for local spatial structure |
| **Transformer Encoder** | Stacked XCiT blocks with residual connections |
| **Regression Head** | MLP predicting normalized particle mass |
| **Classification Head** | MLP predicting particle class logits |

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

- **XCiT**: El-Nouby et al., "XCiT: Cross-Covariance Image Transformers", NeurIPS 2021
- **L2ViT**: Zheng, "Linear attention vision transformers"
- **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021

## Notebook

Full implementation: [`notebooks/linear_attention_vit.ipynb`](notebooks/linear_attention_vit.ipynb)
