"""
Next-Gen Vision Transformer for Particle Physics (CMS Detector Data)
=====================================================================

Implements a multi-task Vision Transformer for CMS jet images that
simultaneously performs:
  - **Quark-Gluon Classification** (CrossEntropyLoss)
  - **Particle Mass Regression** (MSELoss)

Architecture implements **L2ViT** (arXiv:2501.16182):
  "The Linear Attention Resurrection in Vision Transformer", Zheng 2025.

Key contributions from the paper:
  1. **ReLU-based linear attention** — φ(Q)(φ(K)ᵀV) / clamp(φ(Q)·Σφ(K), 1e2)
     achieving O(N·C²) complexity while preserving the non-negative property.
  2. **Local Concentration Module (LCM)** — two depth-wise convolutions
     (DWConv₁ → GELU → BN → DWConv₂, kernel 7×7) that refocus the
     dispersive linear-attention map on local neighbourhoods (paper Eq. 6-7,
     Fig. 5).
  3. **Linear Global Attention (LGA) block** — linear attention followed by
     an LCM residual: Y = LCM(LN(X)) + X  (paper Eq. 8).
  4. **Local Window Attention (LWA) block** — standard multi-head self-
     attention restricted to non-overlapping w×w spatial windows.
  5. **Alternating LGA + LWA** blocks so the model captures both global
     patch-to-patch context and fine-grained local representations.

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


# Minimum value used to clamp the linear-attention denominator.
# A value of 1e2 gives the best accuracy/stability trade-off per L2ViT
# Appendix 8, Table 10 (values below 1e-1 cause NaN; gains plateau above 1e3).
_DENOM_CLAMP_MIN: float = 1e2

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

def build_lcm(embed_dim: int, kernel_size: int = 7) -> nn.Module:
    """Build the L2ViT Local Concentration Module (LCM).

    Linear attention distributes attention scores uniformly over all patches,
    losing the local concentration property of softmax attention (Fig. 1 of
    L2ViT paper).  The LCM re-focuses the output by applying two depth-wise
    convolutional layers that reinforce local spatial interactions.

    Implementation exactly matches the reference code in L2ViT paper Fig. 5
    and Eqs. 6-7::

        X̂     = GELU( DWConv₁(Rearrange(X)) )       # (B, C, H, W)
        X_LCM = Rearrange( DWConv₂( BN( X̂ ) ) )    # (B, N, C)

    Applied as a residual in the LGA block (paper Eq. 8):
        Y = LCM( LN(X) ) + X

    The default kernel size is 7×7 (ablated as optimal in L2ViT Table 6).

    Forward signature: ``lcm(x, H, W) → (B, N, C)``

    Args:
        embed_dim: Channel dimension C.
        kernel_size: Depth-wise conv kernel size (default 7 as in paper).

    Returns:
        nn.Module implementing LCM.  Input/output shape: (B, N, C).
    """

    class _LCM(nn.Module):
        """Local Concentration Module (L2ViT paper Fig. 5)."""

        def __init__(self, dim: int, ks: int) -> None:
            super().__init__()
            padding = ks // 2
            # DWConv₁: depth-wise convolution
            self.conv1 = nn.Conv2d(
                dim, dim, kernel_size=ks, padding=padding, groups=dim
            )
            self.act = nn.GELU()
            # Batch norm between the two convolutions
            self.bn = nn.BatchNorm2d(dim)
            # DWConv₂: second depth-wise convolution
            self.conv2 = nn.Conv2d(
                dim, dim, kernel_size=ks, padding=padding, groups=dim
            )

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            # x: (B, N, C)  →  rearrange to (B, C, H, W) for conv ops
            B, N, C = x.shape
            x = x.transpose(-1, -2).contiguous().view(B, C, H, W)
            x = self.conv1(x)   # DWConv₁
            x = self.act(x)     # GELU  (Eq. 6)
            x = self.bn(x)      # BN    (Eq. 7)
            x = self.conv2(x)   # DWConv₂
            # Reshape back to sequence: (B, C, H, W) → (B, N, C)
            x = x.flatten(2).transpose(-1, -2)
            return x

    return _LCM(embed_dim, kernel_size)


# ---------------------------------------------------------------------------
# 4. Linear Global Attention (LGA) Block
# ---------------------------------------------------------------------------

def build_lga_block(embed_dim: int,
                    num_heads: int,
                    lcm_kernel: int = 7) -> nn.Module:
    """Build an L2ViT Linear Global Attention (LGA) block.

    The LGA block realises the two key ideas of the L2ViT paper:

    **1. ReLU-based linear attention** (Section 4.1, Eq. 3-4):
    Using φ = ReLU as the kernel feature map guarantees that all entries of
    the attention matrix are non-negative (matching the property of softmax)
    while reducing complexity from O(N²C) to O(NC²) by multiplying K and V
    first::

        O = φ(Q) · (φ(K)ᵀ V · s) / clamp(φ(Q) · Σφ(K)ᵀ, min=1e2)

    where s is a learnable scale parameter initialised at √C (per paper
    Appendix 7) and the denominator is clamped to [1e2, +∞) for training
    stability (paper Appendix 8, Table 10).

    **2. Local Concentration Module (LCM)** applied as a residual on the
    linear-attention output (Section 4.2, Eq. 8)::

        Y = LCM( LN(X) ) + X

    Full LGA block recipe (Fig. 3 right)::

        x = CPE(x)                              # conditional positional enc.
        la_out = LA( LN₁(x) )                  # ReLU linear attention
        y = LCM( LN₂(la_out) ) + la_out        # LCM residual  (Eq. 8)
        x = x + y                               # main skip connection
        x = x + FFN( LN₃(x) )                  # feed-forward sub-block

    Args:
        embed_dim: Token feature dimension C.
        num_heads: Number of attention heads.
        lcm_kernel: Kernel size for LCM depth-wise convolutions (default 7).

    Returns:
        nn.Module whose forward signature is ``(x, H, W) -> x``.
    """
    assert embed_dim % num_heads == 0, (
        f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
    )

    class _LGABlock(nn.Module):
        """L2ViT Linear Global Attention block (ReLU-LA + LCM + FFN)."""

        def __init__(self, dim: int, heads: int, ks: int) -> None:
            super().__init__()
            self.heads = heads
            self.d_head = dim // heads

            # --- Linear attention components ---
            # Q, K, V projections
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            # Output projection
            self.proj = nn.Linear(dim, dim)
            # Learnable scale parameter s, initialised to √C (paper App. 7).
            # The paper does not specify per-head scaling; a single shared
            # scalar is sufficient and keeps the implementation clean.
            self.scale = nn.Parameter(
                torch.full((1,), math.sqrt(dim))
            )

            # --- LCM components (Eq. 8) ---
            self.lcm = build_lcm(dim, kernel_size=ks)
            self.ln_lcm = nn.LayerNorm(dim)  # LN before LCM (Eq. 8)

            # --- Block norms ---
            self.ln1 = nn.LayerNorm(dim)   # before linear attention
            self.ln2 = nn.LayerNorm(dim)   # before FFN (accessed by _Block)

        def _linear_attention(
            self, x: torch.Tensor
        ) -> torch.Tensor:
            """ReLU-based linear attention (L2ViT Eq. 3-4).

            Complexity: O(N · C²) — key and value are multiplied first.
            Denominator is clamped to [1e2, +∞) for stability (Table 10).
            """
            B, N, C = x.shape
            # Project to Q, K, V
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, C)

            # Multi-head reshape: (B, heads, N, d_head)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            # φ = ReLU — ensures non-negative attention values (Section 4.1)
            q = F.relu(q)
            k = F.relu(k)

            # φ(K)ᵀ V: shape (B, heads, d_head, d_head)  — O(N·C²) step
            kv = torch.matmul(k.transpose(-2, -1), v) * self.scale

            # Denominator: φ(Q) · Σφ(K)ᵀ → (B, heads, N, 1)
            # Σφ(K) = sum over N tokens → (B, heads, d_head)
            k_sum = k.sum(dim=-2)  # (B, heads, d_head)
            # dot product per token: (B, heads, N, d_head) × (B, heads, d_head, 1)
            denom = torch.matmul(q, k_sum.unsqueeze(-1))  # (B, heads, N, 1)
            # Clamp to [_DENOM_CLAMP_MIN, +∞) for training stability (Appendix 8)
            denom = denom.clamp(min=_DENOM_CLAMP_MIN)

            # Numerator: φ(Q) · (φ(K)ᵀ V) → (B, heads, N, d_head)
            out = torch.matmul(q, kv) / denom  # (B, heads, N, d_head)

            # Merge heads: (B, N, C)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.proj(out)

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            # CPE is applied externally in the transformer block
            # 1. ReLU linear attention (pre-norm)
            la_out = self._linear_attention(self.ln1(x))

            # 2. LCM residual: Y = LCM(LN(la_out)) + la_out  (Eq. 8)
            y = self.lcm(self.ln_lcm(la_out), H, W) + la_out

            # 3. Main skip connection
            return x + y

    return _LGABlock(embed_dim, num_heads, lcm_kernel)


# ---------------------------------------------------------------------------
# 4b. Local Window Attention (LWA) Block
# ---------------------------------------------------------------------------

def build_lwa_block(embed_dim: int,
                    num_heads: int,
                    window_size: int = 4) -> nn.Module:
    """Build an L2ViT Local Window Attention (LWA) block.

    The LWA block applies standard multi-head self-attention within
    non-overlapping spatial windows (Swin-style, L2ViT Fig. 3 left).
    Window attention introduces locality and translational invariance that
    complement the global context captured by the LGA block.

    LWA block recipe (Fig. 3 left)::

        x = CPE(x)                           # conditional positional enc.
        x = x + WA( LN₁(x) )               # window attention + residual
        x = x + FFN( LN₂(x) )              # feed-forward sub-block

    Args:
        embed_dim: Token feature dimension C.
        num_heads: Number of attention heads.
        window_size: Side length w of each square attention window.
            If H or W is not divisible by w the feature map is padded.

    Returns:
        nn.Module whose forward signature is ``(x, H, W) -> x``.
    """
    assert embed_dim % num_heads == 0, (
        f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
    )

    class _LWABlock(nn.Module):
        """L2ViT Local Window Attention block."""

        def __init__(self, dim: int, heads: int, ws: int) -> None:
            super().__init__()
            self.heads = heads
            self.ws = ws
            self.scale = (dim // heads) ** -0.5
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.proj = nn.Linear(dim, dim)
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)   # before FFN (accessed by _Block)

        def _window_attention(
            self, x: torch.Tensor, H: int, W: int
        ) -> torch.Tensor:
            """Multi-head self-attention inside non-overlapping w×w windows."""
            B, N, C = x.shape
            ws = self.ws

            # Reshape tokens to 2-D spatial map: (B, H, W, C)
            x2d = x.view(B, H, W, C)

            # Pad to make H and W divisible by ws
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h > 0 or pad_w > 0:
                x2d = F.pad(x2d, (0, 0, 0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w

            # Partition into windows: (num_windows * B, ws, ws, C)
            nH, nW = Hp // ws, Wp // ws
            x_win = x2d.view(B, nH, ws, nW, ws, C)
            x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous()
            x_win = x_win.view(-1, ws * ws, C)   # (B*nH*nW, ws²,  C)

            # Standard multi-head self-attention within each window
            Bw = x_win.shape[0]
            qkv = self.qkv(x_win)
            q, k, v = qkv.chunk(3, dim=-1)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)                  # (Bw, heads, ws², d)
            out = rearrange(out, 'b h n d -> b n (h d)')  # (Bw, ws², C)
            out = self.proj(out)

            # Reverse windowing: back to (B, H, W, C)
            out = out.view(B, nH, nW, ws, ws, C)
            out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
            out = out.view(B, Hp, Wp, C)
            # Remove padding
            out = out[:, :H, :W, :].contiguous()
            return out.view(B, H * W, C)

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            # CPE is applied externally in the transformer block
            # Window attention (pre-norm + residual)
            return x + self._window_attention(self.ln1(x), H, W)

    return _LWABlock(embed_dim, num_heads, window_size)


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
                             block_type: str = 'lga',
                             mlp_ratio: float = 4.0,
                             drop_rate: float = 0.1,
                             lcm_kernel: int = 7,
                             window_size: int = 4) -> nn.Module:
    """Build one complete L2ViT transformer block (LGA or LWA).

    Both block types share the same outer structure (CPE + attention
    sub-block + FFN) but differ in the attention mechanism:

    * **LGA** (Linear Global Attention, default):
        CPE → ReLU-LA + LCM residual → FFN
    * **LWA** (Local Window Attention):
        CPE → Window self-attention → FFN

    Pre-norm recipe is used throughout::

        x = CPE(x)
        x = x + attention_sublayer( LN₁(x) )    [handled inside LGA/LWA]
        x = x + FFN( LN₂(x) )

    Args:
        embed_dim: Token feature dimension.
        num_heads: Attention heads.
        block_type: ``'lga'`` for Linear Global Attention or
            ``'lwa'`` for Local Window Attention.
        mlp_ratio: FFN hidden-dim multiplier.
        drop_rate: Dropout probability.
        lcm_kernel: LCM depth-wise conv kernel size (LGA only, default 7).
        window_size: Window side length (LWA only, default 4).

    Returns:
        nn.Module for one transformer block. Forward signature:
        ``(x, H, W) -> x``.
    """
    cpe = build_cpe(embed_dim)
    ffn = build_ffn(embed_dim, mlp_ratio, drop_rate)

    if block_type == 'lga':
        attn_block = build_lga_block(embed_dim, num_heads,
                                     lcm_kernel=lcm_kernel)
    elif block_type == 'lwa':
        attn_block = build_lwa_block(embed_dim, num_heads,
                                     window_size=window_size)
    else:
        raise ValueError(
            f"block_type must be 'lga' or 'lwa', got '{block_type}'"
        )

    class _Block(nn.Module):
        """Single L2ViT transformer block (LGA or LWA) with CPE and FFN."""

        def __init__(self) -> None:
            super().__init__()
            self.cpe = cpe
            self.attn = attn_block
            self.ffn = ffn

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            # 1. Conditional Positional Encoding
            x = self.cpe(x, H, W)
            # 2. Attention sub-block (LGA or LWA)
            x = self.attn(x, H, W)
            # 3. FFN sub-block (pre-norm + residual); ln2 lives in self.attn
            x = x + self.ffn(self.attn.ln2(x))
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
    lcm_kernel: int = 7,
    window_size: int = 4,
) -> nn.Module:
    """Build the complete L2ViT-style model for CMS jet images.

    Implements the L2ViT architecture (arXiv:2501.16182) adapted for
    particle-physics detector images.  Blocks alternate between LWA
    (Local Window Attention) and LGA (Linear Global Attention) as described
    in the paper (Section 4.3, Fig. 3): in each pair the LWA first models
    fine-grained short-range interactions, then the LGA builds the global
    patch-to-patch context::

        Input: (B, in_chans, img_size, img_size)
            ↓  ConvStem  (3× stride-2, 3 conv layers)
        Feature map: (B, embed_dim, H', W')
            ↓  Flatten to tokens: (B, H'*W', embed_dim)
        ┌──────────────────────────────────────────────────────────────┐
        │  × (depth // 2)  paired blocks:                              │
        │     [LWA] CPE → Window-Attn → FFN                           │
        │     [LGA] CPE → ReLU-LinAttn + LCM residual → FFN           │
        │  + (depth % 2) extra LGA block if depth is odd              │
        └──────────────────────────────────────────────────────────────┘
            ↓  LayerNorm
            ↓  Global Average Pool  →  (B, embed_dim)
            ├─→ ClassHead  →  (B, num_classes)
            └─→ RegrHead   →  (B, 1)

    Args:
        img_size: Spatial size of the input image (default 125 for CMS).
        in_chans: Number of input channels (3: Tracker, ECAL, HCAL).
        num_classes: Number of classification labels (2: quark vs. gluon).
        embed_dim: Token embedding dimension.
        depth: Total number of transformer blocks.  Alternates LWA, LGA,
            LWA, LGA, … so even values give perfectly paired blocks.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden-dim expansion ratio.
        drop_rate: Dropout probability used in FFN.
        lcm_kernel: Kernel size for LCM depth-wise convolutions (default 7).
        window_size: Side length of LWA attention windows (default 4).

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

    # Alternate LWA (even indices 0,2,4,…) and LGA (odd indices 1,3,5,…)
    # so that local window attention precedes global linear attention in each
    # pair, matching the L2ViT design (paper Section 4.3).
    blocks = nn.ModuleList([
        build_transformer_block(
            embed_dim, num_heads,
            block_type='lwa' if i % 2 == 0 else 'lga',
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            lcm_kernel=lcm_kernel,
            window_size=window_size,
        )
        for i in range(depth)
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
        """L2ViT for CMS jet images — functional wrapper over ModuleDict."""

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

            # 3. Alternating LWA + LGA transformer blocks
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
