"""
Next-Gen Vision Transformer for Particle Physics (CMS Detector Data)
=====================================================================

Implements a multi-task Vision Transformer for CMS jet images that
simultaneously performs:
  - **Quark-Gluon Classification** (CrossEntropyLoss)
  - **Particle Mass Regression** (MSELoss)

Architecture is inspired by:
  - XCiT (arXiv:2106.09681): Cross-Covariance Image Transformer
  - L2ViT: Local Concentration Module for linear attention

Design principle: **purely functional** — no top-level Python ``class``
definitions.  All components are ``nn.Sequential`` / ``nn.Module`` sub-graphs
assembled by builder functions, and the top-level model is an
``nn.ModuleDict``-backed module created via a factory function.

Physics context
---------------
The CMS detector produces 3-channel jet images:
  Channel 0 — Tracker  (charged-particle pT deposits)
  Channel 1 — ECAL     (electromagnetic calorimeter energy)
  Channel 2 — HCAL     (hadronic calorimeter energy)

Each image is 125×125 pixels.  Distinguishing quark-initiated jets from
gluon-initiated jets is a fundamental task in high-energy physics, and
estimating jet mass in the same pass is a useful regression target.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange


# ---------------------------------------------------------------------------
# 1. Convolutional Stem
# ---------------------------------------------------------------------------

def build_conv_stem(in_chans: int = 3, embed_dim: int = 128) -> nn.Sequential:
    """Build a convolutional patch-embedding stem.

    Uses three 3×3 convolutions each with stride 2 and ``same``-style padding
    to progressively halve spatial dimensions while increasing channel depth:
        (B, in_chans, 125, 125)
        → (B, 32, 63, 63)    [stride-2, pad=1  →  ceil(125/2)=63]
        → (B, 64, 32, 32)    [stride-2, pad=1  →  ceil(63/2)=32]
        → (B, embed_dim, 16, 16)  [stride-2, pad=1  →  ceil(32/2)=16]

    Each conv is followed by BatchNorm and GELU activation.  BatchNorm is
    preferable to LayerNorm here because we operate on spatial feature maps
    (NCHW tensors) rather than sequence tokens.

    Args:
        in_chans: Number of input channels (3 for Tracker/ECAL/HCAL).
        embed_dim: Output channel dimension fed into the transformer.

    Returns:
        nn.Sequential stem module.
    """
    stem = nn.Sequential(
        # Stage 1: in_chans → 32
        nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.GELU(),
        # Stage 2: 32 → 64
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.GELU(),
        # Stage 3: 64 → embed_dim
        nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
    )
    return stem


# ---------------------------------------------------------------------------
# 2. Conditional Positional Encoding (CPE)
# ---------------------------------------------------------------------------

def build_cpe(embed_dim: int) -> nn.Module:
    """Build a Conditional Positional Encoding module.

    CPE uses a depth-wise 3×3 convolution applied to the spatial feature map
    and adds the result to the input (residual).  This is position-conditional
    because the convolution kernel "sees" a neighbourhood and implicitly encodes
    relative position.  Unlike sinusoidal or learnable absolute position
    embeddings, CPE generalises to arbitrary spatial resolutions — important
    for detector images that may be re-sampled.

    Applied as: ``x = x + DWConv(x)`` (with x in (B, C, H, W) form).

    Args:
        embed_dim: Channel dimension of the token feature map.

    Returns:
        nn.Module with a single ``dw_conv`` attribute (depth-wise conv).
        The forward method handles the reshape to/from (B, C, H, W).
    """

    class _CPE(nn.Module):
        """Conditional Positional Encoding via depth-wise 3×3 convolution."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dw_conv = nn.Conv2d(
                dim, dim,
                kernel_size=3, padding=1,
                groups=dim,   # depth-wise: one filter per channel
                bias=True,
            )

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            # x: (B, N, C) with N = H*W
            B, N, C = x.shape
            # Reshape to spatial map for the convolution
            feat = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            # Depth-wise conv for local context / position hint
            feat = self.dw_conv(feat)
            # Back to sequence form
            feat = rearrange(feat, 'b c h w -> b (h w) c')
            return x + feat   # residual

    return _CPE(embed_dim)


# ---------------------------------------------------------------------------
# 3. Local Concentration Module (LCM)
# ---------------------------------------------------------------------------

def build_lcm(embed_dim: int) -> nn.Module:
    """Build the L2ViT Local Concentration Module (LCM).

    Linear attention (XCA) computes a global context vector that is identical
    for all tokens — the so-called "distribution concentration" problem.  The
    LCM counteracts this by adding a depth-wise 3×3 convolution branch on top
    of the Value (V) tensor.  This injects local spatial context that breaks
    the uniformity.

    Forward signature: ``lcm(v, H, W) → (B, N, C)``

    Args:
        embed_dim: Channel dimension.

    Returns:
        nn.Module implementing the local-context branch.
    """

    class _LCM(nn.Module):
        """Local Concentration Module: depthwise conv on Value tokens."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.local_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                          groups=dim, bias=False),  # depth-wise
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )

        def forward(self, v: torch.Tensor, H: int, W: int) -> torch.Tensor:
            # v: (B, N, C)
            B, N, C = v.shape
            v_spatial = rearrange(v, 'b (h w) c -> b c h w', h=H, w=W)
            v_local = self.local_conv(v_spatial)
            return rearrange(v_local, 'b c h w -> b (h w) c')

    return _LCM(embed_dim)


# ---------------------------------------------------------------------------
# 4. XCA Block (Cross-Covariance Attention + LCM)
# ---------------------------------------------------------------------------

def build_xca_block(embed_dim: int, num_heads: int) -> nn.Module:
    """Build an XCiT Cross-Covariance Attention block with the LCM attached.

    Standard self-attention is O(N²) in the sequence length N.  XCA instead
    computes attention in the **channel** (head-dimension) space of size
    d_head × d_head, giving O(N) complexity — critical for high-resolution
    detector images.

    The attention matrix is::

        Attn = softmax( (Q̃ᵀ · K̃) / τ )

    where Q̃ = L2-norm(Q) and K̃ = L2-norm(K) are normalised along the token
    dimension (not the feature dimension).  The output is::

        out = V · Attnᵀ   (shape: B, heads, N, d_head)

    then the LCM output is added before projecting back to embed_dim.

    L2 normalisation along the token dimension keeps the dot product bounded
    regardless of sequence length, replacing the 1/√d scaling of vanilla
    attention.

    Temperature τ is a learnable per-head scalar (initialised to 0.2) that
    controls the sharpness of the attention distribution.

    Args:
        embed_dim: Total channel dimension.
        num_heads: Number of attention heads.

    Returns:
        nn.Module implementing XCA + LCM with learned temperature and output
        projection.
    """
    assert embed_dim % num_heads == 0, \
        f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

    class _XCABlock(nn.Module):
        """XCiT Cross-Covariance Attention with Local Concentration Module."""

        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.heads = heads
            self.d_head = dim // heads
            # Project input to Q, K, V
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            # Output projection
            self.proj = nn.Linear(dim, dim)
            # Learnable temperature (one per head)
            # Small initial value makes attention more uniform at start
            self.temperature = nn.Parameter(
                torch.ones(heads, 1, 1) * 0.2
            )
            # Local Concentration Module (L2ViT)
            self.lcm = build_lcm(dim)

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            B, N, C = x.shape
            # Compute Q, K, V  →  (B, N, 3*C)  →  3 × (B, N, C)
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, C)

            # Reshape to multi-head: (B, heads, N, d_head)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            # L2-normalise along the **token** dimension (dim=-2, i.e. N axis)
            # This bounds the dot product and replaces the 1/√d scale factor.
            q = F.normalize(q, p=2, dim=-2)
            k = F.normalize(k, p=2, dim=-2)

            # XCA: attention in channel (d_head) space
            # Attn: (B, heads, d_head, d_head)
            attn = torch.matmul(q.transpose(-2, -1), k) / self.temperature
            attn = F.softmax(attn, dim=-1)

            # Aggregate: (B, heads, N, d_head)
            out = torch.matmul(v, attn.transpose(-2, -1))

            # Merge heads back to (B, N, C)
            out = rearrange(out, 'b h n d -> b n (h d)')

            # LCM: add local spatial context from V (in original token form)
            v_seq = rearrange(v, 'b h n d -> b n (h d)')
            out = out + self.lcm(v_seq, H, W)

            # Final output projection
            return self.proj(out)

    return _XCABlock(embed_dim, num_heads)


# ---------------------------------------------------------------------------
# 5. Feed-Forward Network (FFN)
# ---------------------------------------------------------------------------

def build_ffn(embed_dim: int,
              mlp_ratio: float = 4.0,
              drop_rate: float = 0.1) -> nn.Sequential:
    """Build a two-layer MLP feed-forward network.

    Standard transformer FFN: expand → activate → compress, with dropout after
    each activation for regularisation.

    Args:
        embed_dim: Input/output dimension.
        mlp_ratio: Hidden-dim multiplier (hidden = embed_dim * mlp_ratio).
        drop_rate: Dropout probability.

    Returns:
        nn.Sequential FFN module.
    """
    hidden = int(embed_dim * mlp_ratio)
    return nn.Sequential(
        nn.Linear(embed_dim, hidden),
        nn.GELU(),
        nn.Dropout(drop_rate),
        nn.Linear(hidden, embed_dim),
        nn.Dropout(drop_rate),
    )


# ---------------------------------------------------------------------------
# 6. Transformer Block
# ---------------------------------------------------------------------------

def build_transformer_block(embed_dim: int,
                             num_heads: int,
                             mlp_ratio: float = 4.0,
                             drop_rate: float = 0.1) -> nn.Module:
    """Build one complete transformer block.

    Each block follows the pre-norm recipe::

        x = CPE(x)                             # conditional positional encoding
        x = x + XCA(LN(x))                    # attention sub-block (residual)
        x = x + FFN(LN(x))                    # feed-forward sub-block (residual)

    Args:
        embed_dim: Token feature dimension.
        num_heads: Attention heads.
        mlp_ratio: FFN hidden-dim multiplier.
        drop_rate: Dropout probability.

    Returns:
        nn.Module for one transformer block. Its forward signature is
        ``(x, H, W) -> x`` where H, W are the spatial dimensions.
    """
    cpe = build_cpe(embed_dim)
    xca = build_xca_block(embed_dim, num_heads)
    ffn = build_ffn(embed_dim, mlp_ratio, drop_rate)
    ln1 = nn.LayerNorm(embed_dim)
    ln2 = nn.LayerNorm(embed_dim)

    class _Block(nn.Module):
        """Single XCiT-style transformer block with CPE, XCA+LCM, and FFN."""

        def __init__(self) -> None:
            super().__init__()
            self.cpe = cpe
            self.ln1 = ln1
            self.xca = xca
            self.ln2 = ln2
            self.ffn = ffn

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            # 1. Conditional Positional Encoding
            x = self.cpe(x, H, W)
            # 2. XCA attention sub-block (pre-norm + residual)
            x = x + self.xca(self.ln1(x), H, W)
            # 3. FFN sub-block (pre-norm + residual)
            x = x + self.ffn(self.ln2(x))
            return x

    return _Block()


# ---------------------------------------------------------------------------
# 7. Full Next-Gen ViT Model
# ---------------------------------------------------------------------------

def build_nextgen_vit(
    img_size: int = 125,
    in_chans: int = 3,
    num_classes: int = 2,
    embed_dim: int = 128,
    depth: int = 6,
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.1,
) -> nn.Module:
    """Build the complete Next-Gen Vision Transformer for CMS jet images.

    The model is a single ``nn.Module`` backed by an ``nn.ModuleDict`` of
    named sub-modules (no top-level class definition).  All components are
    assembled from builder functions that return ``nn.Sequential`` /
    ``nn.Module`` instances.

    Architecture overview::

        Input: (B, in_chans, img_size, img_size)
            ↓  ConvStem  (3× stride-2)
        Feature map: (B, embed_dim, H', W')
            ↓  Flatten to tokens: (B, H'*W', embed_dim)
        ┌─────────────────────────────────────────┐
        │  × depth  TransformerBlock               │
        │     CPE → XCA+LCM → FFN                 │
        └─────────────────────────────────────────┘
            ↓  LayerNorm
            ↓  Global Average Pool  →  (B, embed_dim)
            ├─→ ClassHead  →  (B, num_classes)
            └─→ RegrHead   →  (B, 1)

    Args:
        img_size: Spatial size of the input image (default 125 for CMS).
        in_chans: Number of input channels (3: Tracker, ECAL, HCAL).
        num_classes: Number of classification labels (2: quark vs. gluon).
        embed_dim: Token embedding dimension.
        depth: Number of stacked transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden-dim expansion ratio.
        drop_rate: Dropout probability used in FFN.

    Returns:
        nn.Module with a ``forward(x)`` method that returns a dict
        ``{'cls': (B, num_classes), 'reg': (B, 1)}``.
    """
    # Compute spatial size after the 3-stage stride-2 convolutional stem.
    # With padding=1 and stride=2: out = ceil(in / 2)
    H = math.ceil(math.ceil(math.ceil(img_size / 2) / 2) / 2)
    W = H  # square images

    # --- Sub-module construction ---
    stem = build_conv_stem(in_chans, embed_dim)
    blocks = nn.ModuleList([
        build_transformer_block(embed_dim, num_heads, mlp_ratio, drop_rate)
        for _ in range(depth)
    ])
    norm = nn.LayerNorm(embed_dim)
    cls_head = nn.Linear(embed_dim, num_classes)
    reg_head = nn.Linear(embed_dim, 1)

    # Pack into a ModuleDict so all parameters are tracked
    modules = nn.ModuleDict({
        'stem': stem,
        'blocks': blocks,
        'norm': norm,
        'cls_head': cls_head,
        'reg_head': reg_head,
    })

    # --- Model wrapper (single internal class, not exported) ---
    class _NextGenViT(nn.Module):
        """Next-Gen Vision Transformer — functional wrapper over ModuleDict."""

        def __init__(self, mods: nn.ModuleDict, h: int, w: int) -> None:
            super().__init__()
            self.mods = mods
            self.H = h
            self.W = w

        def forward(self, x: torch.Tensor) -> dict:
            """Forward pass.

            Args:
                x: Input tensor of shape (B, in_chans, img_size, img_size).

            Returns:
                dict with keys:
                  'cls' — classification logits (B, num_classes)
                  'reg' — regression output (B, 1)
            """
            # 1. Convolutional Stem: spatial feature extraction
            feat = self.mods['stem'](x)         # (B, C, H, W)

            # 2. Flatten spatial dims to token sequence
            tokens = rearrange(feat, 'b c h w -> b (h w) c')  # (B, N, C)

            # 3. Transformer blocks (CPE + XCA+LCM + FFN)
            for blk in self.mods['blocks']:
                tokens = blk(tokens, self.H, self.W)

            # 4. Final normalisation
            tokens = self.mods['norm'](tokens)  # (B, N, C)

            # 5. Global Average Pooling over token sequence
            pooled = tokens.mean(dim=1)         # (B, C)

            # 6. Multi-task output heads
            cls_out = self.mods['cls_head'](pooled)   # (B, num_classes)
            reg_out = self.mods['reg_head'](pooled)   # (B, 1)

            return {'cls': cls_out, 'reg': reg_out}

    model = _NextGenViT(modules, H, W)
    return model


# ---------------------------------------------------------------------------
# 8. Multi-Task Loss
# ---------------------------------------------------------------------------

def multi_task_loss(
    cls_logits: torch.Tensor,
    reg_output: torch.Tensor,
    cls_target: torch.Tensor,
    reg_target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the combined multi-task loss.

    The total loss is a weighted sum of:
      - Cross-Entropy for Quark/Gluon classification.
      - Mean-Squared-Error for particle mass regression.

    ``total_loss = alpha * CE(cls_logits, cls_target)
                 + beta  * MSE(reg_output, reg_target)``

    Args:
        cls_logits: Predicted class logits, shape (B, num_classes).
        reg_output: Predicted mass, shape (B, 1).
        cls_target: Ground-truth class indices, shape (B,).
        reg_target: Ground-truth mass values, shape (B, 1) or (B,).
        alpha: Weight for the classification loss.
        beta: Weight for the regression loss.

    Returns:
        Tuple of (total_loss, cls_loss, reg_loss) — all scalar tensors.
    """
    ce_loss = F.cross_entropy(cls_logits, cls_target)
    mse_loss = F.mse_loss(reg_output, reg_target.view_as(reg_output))
    total = alpha * ce_loss + beta * mse_loss
    return total, ce_loss, mse_loss


# ---------------------------------------------------------------------------
# 9. Dummy Data Loader
# ---------------------------------------------------------------------------

def create_dummy_dataloader(
    num_samples: int = 512,
    batch_size: int = 32,
    img_size: int = 125,
    in_chans: int = 3,
    num_classes: int = 2,
) -> DataLoader:
    """Create a DataLoader of synthetic CMS detector jet images.

    In a real experiment the images would be loaded from HDF5 files
    (as done in the existing notebook).  Here we generate random tensors
    with the correct shapes so the training loop can be exercised without
    an external dataset.

    Channels represent:
      0 — Tracker  (charged-particle transverse-momentum deposits)
      1 — ECAL     (electromagnetic calorimeter)
      2 — HCAL     (hadronic calorimeter)

    Args:
        num_samples: Number of synthetic jet images.
        batch_size: DataLoader mini-batch size.
        img_size: Spatial size of each image (default 125).
        in_chans: Number of detector channels (default 3).
        num_classes: Number of classification labels.

    Returns:
        torch.utils.data.DataLoader yielding (images, cls_labels, reg_targets).
    """
    images = torch.randn(num_samples, in_chans, img_size, img_size)
    cls_labels = torch.randint(0, num_classes, (num_samples,))
    # Regression target: jet mass in GeV (here a random value in [0, 200] GeV)
    reg_targets = torch.rand(num_samples, 1) * 200.0

    dataset = TensorDataset(images, cls_labels, reg_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True)
    return loader


# ---------------------------------------------------------------------------
# 10. Single-Epoch Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> dict:
    """Run one full pass over the training data.

    Args:
        model: The Next-Gen ViT model.
        loader: DataLoader yielding (images, cls_labels, reg_targets).
        optimizer: Gradient-based optimiser.
        device: Compute device (CPU / CUDA).
        alpha: Classification loss weight.
        beta: Regression loss weight.

    Returns:
        dict with keys 'total_loss', 'cls_loss', 'reg_loss', 'accuracy'
        — all averaged over the epoch.
    """
    model.train()
    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    reg_loss_sum = 0.0
    correct = 0
    total_samples = 0

    for imgs, cls_tgt, reg_tgt in loader:
        imgs = imgs.to(device)
        cls_tgt = cls_tgt.to(device)
        reg_tgt = reg_tgt.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss, cls_l, reg_l = multi_task_loss(
            outputs['cls'], outputs['reg'],
            cls_tgt, reg_tgt,
            alpha=alpha, beta=beta,
        )
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss_sum += loss.item() * bs
        cls_loss_sum += cls_l.item() * bs
        reg_loss_sum += reg_l.item() * bs

        preds = outputs['cls'].argmax(dim=1)
        correct += (preds == cls_tgt).sum().item()
        total_samples += bs

    n = total_samples
    return {
        'total_loss': total_loss_sum / n,
        'cls_loss': cls_loss_sum / n,
        'reg_loss': reg_loss_sum / n,
        'accuracy': correct / n,
    }


# ---------------------------------------------------------------------------
# 11. Main Training Script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ---- Hyperparameters ----
    IMG_SIZE = 125
    IN_CHANS = 3
    NUM_CLASSES = 2
    EMBED_DIM = 128
    DEPTH = 6
    NUM_HEADS = 4
    MLP_RATIO = 4.0
    DROP_RATE = 0.1
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LR = 1e-3
    ALPHA = 1.0   # weight for classification loss
    BETA = 0.5    # weight for regression loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Build model ----
    model = build_nextgen_vit(
        img_size=IMG_SIZE,
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        drop_rate=DROP_RATE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---- Data ----
    train_loader = create_dummy_dataloader(
        num_samples=512,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
    )
    print(f"Training batches per epoch: {len(train_loader)}")

    # ---- Optimiser + Scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    # ---- Training loop ----
    print("\n{'Epoch':>6}  {'Total':>8}  {'CE':>8}  {'MSE':>8}  {'Acc':>7}")
    print('-' * 50)
    for epoch in range(1, NUM_EPOCHS + 1):
        metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            alpha=ALPHA, beta=BETA,
        )
        scheduler.step()
        print(
            f"{epoch:>6}  "
            f"{metrics['total_loss']:>8.4f}  "
            f"{metrics['cls_loss']:>8.4f}  "
            f"{metrics['reg_loss']:>8.4f}  "
            f"{metrics['accuracy']:>7.3%}"
        )

    print("\nTraining complete.")
