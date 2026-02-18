"""
U-Net for semantic segmentation using a MoCo v3 pretrained backbone.

Architecture overview
─────────────────────
ResNet50 encoder  → hierarchical 5-scale features → classic U-Net decoder
                    with lateral skip connections from each ResNet stage.

ViT-Small / ViT-Base encoder → patch-grid features from 4 equally-spaced
                    transformer blocks → SETR-style decoder that progressively
                    doubles spatial resolution, using each intermediate block's
                    feature map as a skip connection (bilinearly upsampled to
                    the current decoder resolution before concatenation).

Channel layout — ResNet50 (224×224 input)
──────────────────────────────────────────
  Encoder stage    Channels   Spatial
  ─────────────    ────────   ───────
  stem             64         112×112   ← skip s0 (before maxpool)
  layer1           256        56×56     ← skip s1
  layer2           512        28×28     ← skip s2
  layer3           1024       14×14     ← skip s3
  layer4 (bot.)    2048       7×7

  Decoder step     In-ch      Skip-ch   Out-ch   Spatial
  ─────────────    ─────      ───────   ──────   ───────
  up1              2048       1024      512      14×14
  up2              512        512       256      28×28
  up3              256        256       128      56×56
  up4              128        64        64       112×112
  up5              64         —         32       224×224 (no skip)
  head             32         —         C        224×224

Channel layout — ViT (224×224 input, patch=16, depth=12, embed_dim=D)
──────────────────────────────────────────────────────────────────────
  Encoder segment    Channels   Spatial
  ───────────────    ────────   ───────
  blocks 0–2   (s0)   D        14×14
  blocks 3–5   (s1)   D        14×14
  blocks 6–8   (s2)   D        14×14
  blocks 9–11  (bot.) D        14×14

  Decoder step     In-ch        Skip-ch   Out-ch   Spatial
  ─────────────    ─────        ───────   ──────   ───────
  up1              D            D//4      D//2     28×28
  up2              D//2         D//4      D//4     56×56
  up3              D//4         D//4      D//8     112×112
  up4              D//8         —         D//16    224×224 (no skip)
  head             D//16        —         C        224×224

  For ViT-Small: D=384  → D//2=192, D//4=96, D//8=48, D//16=24
  For ViT-Base:  D=768  → D//2=384, D//4=192, D//8=96, D//16=48

Usage
─────
  from src.models.unet import UNet

  model = UNet(
      backbone_name       = "resnet50",             # "vit_small" | "vit_base"
      num_classes         = 1,                      # 1 for binary segmentation
      checkpoint_path     = "checkpoints/best.pth", # MoCo v3 pretrained checkpoint
      freeze_encoder      = False,
  )
  logits = model(x)   # [B, num_classes, H, W]  raw logits, no sigmoid/softmax
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import get_backbone


# ─── Shared building blocks ───────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    """3×3 Conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """
    Single U-Net decoder step.

        1. Bilinear 2× upsample of the incoming feature map.
        2. (Optional) concatenate a skip connection along the channel axis.
           The skip is bilinearly resized to the upsampled spatial size if
           there is a boundary mismatch (e.g. odd input dimensions).
        3. Two ConvBnRelu layers to mix and compress channels.

    When skip_ch=0 and skip=None, the block acts as a plain upsampling block
    (the conv input is in_ch, not in_ch + 0).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(
                    skip, size=x.shape[-2:], mode='bilinear', align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── ResNet encoder ───────────────────────────────────────────────────────────

class ResNetEncoder(nn.Module):
    """
    Wraps a torchvision ResNet50 (fc replaced with Identity) and extracts 5
    feature maps at different spatial scales for the U-Net decoder.

    The backbone is stored as ``self.backbone`` so its state-dict keys remain
    identical to those saved in a MoCo v3 checkpoint (under the 'base_model.*'
    prefix), making weight loading straightforward.

    Outputs for a 224×224 input:
        skips[0]  [B, 64,   112, 112]  — stem: conv1 + bn1 + relu (no pool)
        skips[1]  [B, 256,  56,  56]   — layer1
        skips[2]  [B, 512,  28,  28]   — layer2
        skips[3]  [B, 1024, 14,  14]   — layer3
        out       [B, 2048, 7,   7]    — layer4  (bottleneck)
    """

    #: Channel widths of [s0, s1, s2, s3, bottleneck]
    channel_dims = (64, 256, 512, 1024, 2048)

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone  # torchvision ResNet50 with fc = Identity

    def forward(self, x: torch.Tensor):
        b = self.backbone
        # Stem — intentionally stop before maxpool to keep H/2 skip at s0
        s0  = b.relu(b.bn1(b.conv1(x)))   # [B, 64,   H/2,  W/2]
        x   = b.maxpool(s0)               # [B, 64,   H/4,  W/4]
        s1  = b.layer1(x)                 # [B, 256,  H/4,  W/4]
        s2  = b.layer2(s1)                # [B, 512,  H/8,  W/8]
        s3  = b.layer3(s2)                # [B, 1024, H/16, W/16]
        out = b.layer4(s3)                # [B, 2048, H/32, W/32]
        return [s0, s1, s2, s3], out


# ─── ViT encoder ──────────────────────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    Wraps a timm ViT (vit_small_patch16_224 or vit_base_patch16_224) and
    exposes intermediate patch-token feature maps at 4 equally-spaced depths.

    The transformer blocks are divided into 4 equal-length segments.  After
    each of the first 3 segments the patch tokens are reshaped from
    [B, N, D] → [B, D, h_p, w_p] and stored as skip connections. The output
    of the final segment (after layer norm) forms the bottleneck.

    For 224×224 input with patch_size=16: h_p = w_p = 14.
    All skips and the bottleneck share the same spatial resolution.

    Outputs:
        skips[0]  [B, D, 14, 14]  — after blocks   0 … depth//4-1
        skips[1]  [B, D, 14, 14]  — after blocks depth//4 … depth//2-1
        skips[2]  [B, D, 14, 14]  — after blocks depth//2 … 3*depth//4-1
        out       [B, D, 14, 14]  — after blocks 3*depth//4 … depth-1 + norm
    """

    def __init__(self, backbone: nn.Module, image_size: int = 224) -> None:
        super().__init__()
        self.backbone = backbone

        depth = len(backbone.blocks)
        D = backbone.embed_dim
        ps = backbone.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            ps = ps[0]

        self.h_p = image_size // ps
        self.w_p = image_size // ps
        self.D   = D

        # Indices at which to snapshot intermediate features (not including last)
        self._cut_set = {depth // 4, depth // 2, 3 * depth // 4}

    @property
    def channel_dims(self):
        D = self.D
        return (D, D, D, D, D)

    def _to_map(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1+N, D] → [B, D, h_p, w_p]  (drops CLS token and reshapes)."""
        B = x.shape[0]
        patches = x[:, 1:].transpose(1, 2)          # [B, D, N]
        return patches.reshape(B, self.D, self.h_p, self.w_p)

    def forward(self, x: torch.Tensor):
        b = self.backbone

        # Patch embedding + CLS token + positional encoding
        x = b.patch_embed(x)
        B = x.shape[0]
        cls = b.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + b.pos_embed
        # pos_drop: present in timm < 0.9; absent in timm >= 1.0 (dropped in favour of
        # patch_drop + norm_pre handled below).
        if hasattr(b, 'pos_drop'):
            x = b.pos_drop(x)
        # patch_drop / norm_pre: added in timm 1.x.  Both are no-ops with default
        # model creation (patch_drop_rate=0, pre_norm=False), but we apply them
        # here so the manual forward stays correct if those flags are ever set.
        if hasattr(b, 'patch_drop'):
            x = b.patch_drop(x)
        if hasattr(b, 'norm_pre'):
            x = b.norm_pre(x)

        skips = []
        for idx, block in enumerate(b.blocks):
            x = block(x)
            if (idx + 1) in self._cut_set:
                skips.append(self._to_map(x))   # snapshot without final norm

        out = self._to_map(b.norm(x))            # apply norm only to bottleneck
        return skips, out                        # 3 skips + bottleneck


# ─── ResNet decoder ───────────────────────────────────────────────────────────

class ResNetDecoder(nn.Module):
    """Classic U-Net decoder matched to ResNetEncoder's 5-scale feature maps."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # channel_dims = (64, 256, 512, 1024, 2048)
        self.up1 = DecoderBlock(2048, 1024, 512)   # 7→14
        self.up2 = DecoderBlock(512,  512,  256)   # 14→28
        self.up3 = DecoderBlock(256,  256,  128)   # 28→56
        self.up4 = DecoderBlock(128,  64,   64)    # 56→112
        # No skip at full resolution: simple upsample + conv
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBnRelu(64, 32),
            ConvBnRelu(32, 32),
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(
        self,
        skips: list,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        s0, s1, s2, s3 = skips
        x = self.up1(bottleneck, s3)   # 512  @ H/16
        x = self.up2(x, s2)            # 256  @ H/8
        x = self.up3(x, s1)            # 128  @ H/4
        x = self.up4(x, s0)            # 64   @ H/2
        x = self.up5(x)                # 32   @ H
        return self.head(x)            # C    @ H


# ─── ViT decoder ──────────────────────────────────────────────────────────────

class ViTDecoder(nn.Module):
    """
    SETR-style progressive decoder for ViT encoders.

    Since all ViT skip connections are at the same spatial resolution (14×14),
    each skip is projected to D//4 channels and then bilinearly upsampled to
    match the decoder's current spatial resolution before concatenation.

    The decoder produces output at the full input resolution (224×224 for a
    224×224 ViT input with patch_size=16).
    """

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        D = embed_dim

        # 1×1 convolutions that compress each skip from D → D//4 channels
        # before injecting into the decoder (reduces memory and computation)
        self.proj2 = nn.Conv2d(D, D // 4, kernel_size=1)   # for skip s2
        self.proj1 = nn.Conv2d(D, D // 4, kernel_size=1)   # for skip s1
        self.proj0 = nn.Conv2d(D, D // 4, kernel_size=1)   # for skip s0

        # Decoder blocks (each internally does 2× upsample then concat then conv)
        self.up1 = DecoderBlock(D,       D // 4, D // 2)   # 14→28
        self.up2 = DecoderBlock(D // 2,  D // 4, D // 4)   # 28→56
        self.up3 = DecoderBlock(D // 4,  D // 4, D // 8)   # 56→112
        self.up4 = DecoderBlock(D // 8,  0,      D // 16)  # 112→224 (no skip)

        self.head = nn.Conv2d(D // 16, num_classes, kernel_size=1)

    def forward(
        self,
        skips: list,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        s0, s1, s2 = skips                          # all [B, D, 14, 14]

        x = self.up1(bottleneck, self.proj2(s2))    # D//2  @ 28×28
        x = self.up2(x,         self.proj1(s1))     # D//4  @ 56×56
        x = self.up3(x,         self.proj0(s0))     # D//8  @ 112×112
        x = self.up4(x)                             # D//16 @ 224×224
        return self.head(x)                         # C     @ 224×224


# ─── Top-level UNet ───────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Encoder–decoder segmentation network backed by a MoCo v3 pretrained
    ResNet50 or ViT backbone.

    Args:
        backbone_name:      "resnet50", "vit_small", or "vit_base".
        num_classes:        Number of output channels (1 for binary segmentation).
        checkpoint_path:    Path to a MoCo v3 ``best.pth`` / ``last.pth``
                            checkpoint.  Only the ``base_model.*`` (query
                            encoder backbone) weights are transferred; the
                            projector, predictor and momentum encoder weights
                            are ignored.  Pass None or "" for random init.
        freeze_encoder:     If True, all encoder parameters are frozen
                            (requires_grad=False).  Useful for a linear-probe
                            evaluation or a first warm-up stage.
        image_size:         Expected spatial input size (default 224).
                            Must match the ViT's positional embedding size.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 1,
        checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        # Instantiate backbone (no pretrained weights — we load MoCo ckpt below)
        backbone, _ = get_backbone(backbone_name, pretrained=False)

        if backbone_name == 'resnet50':
            self.encoder = ResNetEncoder(backbone)
            self.decoder = ResNetDecoder(num_classes)
        elif backbone_name in ('vit_small', 'vit_base'):
            self.encoder = ViTEncoder(backbone, image_size=image_size)
            self.decoder = ViTDecoder(self.encoder.D, num_classes)
        else:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Supported: 'resnet50', 'vit_small', 'vit_base'."
            )

        # Load MoCo v3 pretrained encoder weights if a checkpoint is provided
        if checkpoint_path:
            self._load_moco_weights(checkpoint_path)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print(f"=> encoder frozen ({sum(1 for _ in self.encoder.parameters())} params)")
        self._encoder_frozen = freeze_encoder

    def train(self, mode: bool = True) -> "UNet":
        """
        Override to keep a frozen encoder permanently in eval mode.

        Without this, ``model.train()`` in the training loop recursively sets
        every sub-module — including frozen BatchNorm layers inside ResNet50 —
        to train mode.  That causes BN running mean/var to drift from the
        fine-tuning batch statistics, corrupting the pretrained statistics even
        though the weights are frozen.
        """
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()   # keep BN in eval; freeze running stats
        return self

    # ── weight loading ────────────────────────────────────────────────────────

    def _load_moco_weights(self, checkpoint_path: str) -> None:
        """
        Transfer the query encoder backbone weights from a MoCo v3 checkpoint
        into ``self.encoder.backbone``.

        MoCo v3 checkpoints store:
            base_model.*     → query encoder backbone   ← we load this
            projector_q.*    → query projector           (ignored)
            predictor.*      → predictor MLP             (ignored)
            base_model_k.*   → momentum encoder          (ignored)
            projector_k.*    → momentum projector        (ignored)

        The loader handles:
            • DDP 'module.' prefix stripping
            • partial loading (strict=False) with informative output
        """
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            print(
                f"=> no MoCo checkpoint found at '{checkpoint_path}', "
                f"using random initialisation"
            )
            return

        print(f"=> loading MoCo v3 encoder weights from '{checkpoint_path}'")
        ckpt  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt)

        # Strip DDP 'module.' prefix added by DistributedDataParallel
        state = {
            (k[len('module.'):] if k.startswith('module.') else k): v
            for k, v in state.items()
        }

        # Extract only the query encoder backbone weights
        encoder_keys = {
            k[len('base_model.'):]: v
            for k, v in state.items()
            if k.startswith('base_model.')
        }

        if not encoder_keys:
            print(
                "=> Warning: no 'base_model.*' keys found in checkpoint. "
                "The checkpoint may be in an unexpected format. "
                "Falling back to random initialisation."
            )
            return

        missing, unexpected = self.encoder.backbone.load_state_dict(
            encoder_keys, strict=False
        )

        loaded = len(encoder_keys) - len(missing)
        print(
            f"=> transferred {loaded}/{len(encoder_keys)} encoder backbone "
            f"parameters from MoCo checkpoint"
        )
        if missing:
            print(f"   missing ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"   unexpected ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]  input image batch

        Returns:
            logits: [B, num_classes, H, W]  raw logits (no sigmoid/softmax)
                    Apply torch.sigmoid for binary segmentation predictions.
        """
        skips, bottleneck = self.encoder(x)
        return self.decoder(skips, bottleneck)
