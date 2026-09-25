"""
ByteNet: dual-branch (byte + image) classification network.

Reproduces the architecture in Sec. III-C/D of the paper:
  - Byte Branch Feature Extraction (BBFE): a single fully-connected layer
    over the raw 1D byte sequence (Eq. 5) that memorizes co-occurring
    "magic bytes".
  - Image Branch Feature Extraction (IBFE): an embedding layer (Fig. 6)
    followed by a 4-stage hierarchical residual visual feature extractor
    (RVFE) over the Byte2Image representation (Eqs. 6-8).
  - Two RVFE block variants, giving two full-network variants:
      * ByteResNet - n-gram embedding (wide conv) + ResNet blocks (Eq. 13)
      * ByteFormer - patch embedding + PoolFormer blocks (Eqs. 17-18)
  - Feature fusion: concatenate byte-branch and image-branch features,
    then a final FC + softmax (Eq. 9).

Channel widths / block counts are reduced from the paper's full-scale
settings (e.g. ResNet [64,128,256,512], PoolFormer [64,128,320,512] with
depths [6,6,18,6]) so the network trains in minutes on CPU; the
*structure* (branches, stages, fusion) is unchanged. See README for the
paper's original hyperparameters and how to restore them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Byte Branch Feature Extraction (BBFE) -- Eq. (5)
# --------------------------------------------------------------------------
class ByteBranch(nn.Module):
    """Shallow FC layer over the raw normalized byte sequence."""

    def __init__(self, sector_size: int, out_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(sector_size, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, raw_bytes: torch.Tensor) -> torch.Tensor:
        # raw_bytes: (B, Ns) float in [0, 1]
        return self.act(self.fc(raw_bytes))


# --------------------------------------------------------------------------
# Embedding layers (Fig. 6)
# --------------------------------------------------------------------------
class NgramEmbeddingSimple(nn.Module):
    """N-gram embedding used by ByteResNet (Eqs. 10-12).

    A wide convolution (kernel width == image width) turns each row of
    the Byte2Image matrix into K scalar embeddings per row; the K
    resulting values per row are stacked into a (H, K) map ("x_emb" in
    the paper), then a 7x7 conv + pooling project this into the
    (H', W', C1) input of stage 1.
    """

    def __init__(self, img_h: int, img_w: int, k: int = 32, c1: int = 32):
        super().__init__()
        self.k = k
        self.wide_conv = nn.Conv2d(1, k, kernel_size=(1, img_w))
        self.proj = nn.Conv2d(1, c1, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x_im: torch.Tensor) -> torch.Tensor:
        # x_im: (B, 1, H, W) -> wide_conv -> (B, K, H, 1)
        x_emb = self.wide_conv(x_im)          # (B, K, H, 1)
        x_emb = x_emb.squeeze(-1)             # (B, K, H)
        x_emb = x_emb.permute(0, 2, 1)        # (B, H, K)
        x_emb = x_emb.unsqueeze(1)            # (B, 1, H, K)  == x_emb in paper (Eq. 11)
        x0 = self.proj(x_emb)                 # (B, C1, H, K)
        x0 = self.pool(self.act(self.bn(x0)))
        return x0


class PatchEmbedding(nn.Module):
    """Patch embedding used by ByteFormer (Eqs. 14-16)."""

    def __init__(self, img_h: int, img_w: int, patch: int = 8, c1: int = 32):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(1, c1, kernel_size=patch, stride=patch)
        self.pos = None  # lazily-created learned positional embedding
        self.c1 = c1

    def forward(self, x_im: torch.Tensor) -> torch.Tensor:
        x0 = self.proj(x_im)  # (B, C1, H', W')
        if self.pos is None or self.pos.shape[-2:] != x0.shape[-2:]:
            self.pos = nn.Parameter(torch.zeros(1, self.c1, *x0.shape[-2:]).to(x0.device))
        return x0 + self.pos


# --------------------------------------------------------------------------
# RVFE feature-extraction blocks
# --------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    """Classic residual block, Eq. (13)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class PoolFormerBlock(nn.Module):
    """PoolFormer block, Eqs. (17-18): pooling token-mixer + channel MLP."""

    def __init__(self, channels: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        z = (self.pool(self.norm1(x)) - self.norm1(x)) + x  # pooling token mixer (subtract identity, as in PoolFormer)
        y = self.mlp(self.norm2(z)) + z
        return y


class RVFEStage(nn.Module):
    """One stage = L feature-extraction blocks + downsampling conv."""

    def __init__(self, in_ch, out_ch, depth, block_type="resnet", downsample=True, is_last=False):
        super().__init__()
        block_cls = ResNetBlock if block_type == "resnet" else PoolFormerBlock
        self.in_proj = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.blocks = nn.Sequential(*[block_cls(out_ch) for _ in range(depth)])
        self.is_last = is_last
        if is_last:
            self.down = nn.AdaptiveAvgPool2d(1)  # GAP, Eq. (8)
        elif downsample:
            self.down = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1), nn.ReLU(inplace=True)
            )
        else:
            self.down = nn.Identity()

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.down(x)
        return x


# --------------------------------------------------------------------------
# Full ByteNet
# --------------------------------------------------------------------------
class ByteNet(nn.Module):
    """Dual-branch ByteNet (ByteResNet or ByteFormer variant).

    Args:
        sector_size: Ns, raw fragment length in bytes.
        img_h, img_w: Byte2Image output size (H = Ns-n+1, W = 8n).
        num_classes: number of file-type classes.
        variant: 'resnet' or 'poolformer'.
        stage_channels: channel width per of the 4 RVFE stages.
        stage_depths: number of blocks per stage.
        byte_feat_dim: BBFE output dimension F0.
    """

    def __init__(
        self,
        sector_size: int,
        img_h: int,
        img_w: int,
        num_classes: int,
        variant: str = "resnet",
        stage_channels=(32, 64, 128, 256),
        stage_depths=(1, 1, 1, 1),
        byte_feat_dim: int = 128,
        ngram_k: int = 32,
        patch_size: int = 8,
    ):
        super().__init__()
        assert variant in ("resnet", "poolformer")
        self.variant = variant

        self.byte_branch = ByteBranch(sector_size, out_dim=byte_feat_dim)

        c1 = stage_channels[0]
        if variant == "resnet":
            self.embed = NgramEmbeddingSimple(img_h, img_w, k=ngram_k, c1=c1)
        else:
            self.embed = PatchEmbedding(img_h, img_w, patch=patch_size, c1=c1)

        block_type = "resnet" if variant == "resnet" else "poolformer"
        stages = []
        in_ch = c1
        for i, (ch, depth) in enumerate(zip(stage_channels, stage_depths)):
            is_last = i == len(stage_channels) - 1
            stages.append(RVFEStage(in_ch, ch, depth, block_type=block_type, is_last=is_last))
            in_ch = ch
        self.stages = nn.ModuleList(stages)

        fused_dim = byte_feat_dim + stage_channels[-1]
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, raw_bytes: torch.Tensor, img: torch.Tensor):
        # raw_bytes: (B, Ns) in [0,1]; img: (B, 1, H, W) in [0,1]
        x_sf = self.byte_branch(raw_bytes)

        x = self.embed(img)
        for stage in self.stages:
            x = stage(x)
        x_df = x.flatten(1)  # after GAP: (B, C4, 1, 1) -> (B, C4)

        fused = torch.cat([x_sf, x_df], dim=1)
        logits = self.classifier(fused)
        return logits

    def forward_byte_only(self, raw_bytes):
        x_sf = self.byte_branch(raw_bytes)
        return self.classifier(torch.cat([x_sf, torch.zeros(
            x_sf.shape[0], self.classifier.in_features - x_sf.shape[1], device=x_sf.device
        )], dim=1))


if __name__ == "__main__":
    # quick shape check
    import numpy as np
    from bytenet.byte2image import byte2image

    Ns, n = 512, 4
    H, W = Ns - n + 1, 8 * n
    model = ByteNet(Ns, H, W, num_classes=7, variant="resnet")
    raw = torch.rand(4, Ns)
    img = torch.rand(4, 1, H, W)
    out = model(raw, img)
    print("ByteResNet output:", out.shape)

    model2 = ByteNet(Ns, H, W, num_classes=7, variant="poolformer")
    out2 = model2(raw, img)
    print("ByteFormer output:", out2.shape)
