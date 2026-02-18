"""
Backbone architectures for MoCo v3.
Supports ResNet-50 and Vision Transformers (via timm).

ViT output dimensions:
    vit_small  -> 384
    vit_base   -> 768

Requires: pip install timm
"""

import torch.nn as nn
import timm
from torchvision import models

def get_backbone(name: str = "resnet50", pretrained: bool = False):
    """
    Returns (model, dim_in) where dim_in is the backbone's output feature dimension.

    Args:
        name:      One of "resnet50", "vit_small", "vit_base".
        pretrained: Whether to load ImageNet-pretrained weights.
    """
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        dim_in = model.fc.in_features   # 2048
        model.fc = nn.Identity()
        return model, dim_in

    elif name == "vit_small":
        # ViT-Small/16: 21M params, embed_dim=384
        # num_classes=0 removes the classification head; model returns [B, 384]
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        dim_in = model.embed_dim
        return model, dim_in

    elif name == "vit_base":
        # ViT-Base/16: 86M params, embed_dim=768
        # num_classes=0 removes the classification head; model returns [B, 768]
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        dim_in = model.embed_dim
        return model, dim_in

    else:
        raise ValueError(
            f"Unknown backbone: '{name}'. "
            f"Supported options: 'resnet50', 'vit_small', 'vit_base'."
        )
