"""
Backbone architectures for MoCo v3.
Supports ResNet-50 and Vision Transformers.
"""

import torch.nn as nn
from torchvision import models

def get_backbone(name: str = "resnet50", pretrained: bool = False):
    """
    Returns the backbone model and its output dimension.
    """
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # Remove the last fully connected layer
        dim_in = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim_in
    elif name.startswith("vit"):
        if name == "vit_small":
            # torchvision doesn't have "small", but we can use vit_b_16 or similar
            # or define a custom small ViT. For simplicity, let's use vit_b_16
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            dim_in = model.heads.head.in_features
            model.heads = nn.Identity()
            return model, dim_in
        elif name == "vit_base":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            dim_in = model.heads.head.in_features
            model.heads = nn.Identity()
            return model, dim_in
        else:
            raise NotImplementedError(f"Backbone {name} not implemented")
    else:
        raise ValueError(f"Unknown backbone: {name}")
