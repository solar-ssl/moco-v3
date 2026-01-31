"""
Backbone architectures for MoCo v3.
Supports ResNet-50 and Vision Transformers.
"""

import torch.nn as nn
from torchvision import models

def get_backbone(name: str = "resnet50", pretrained: bool = False, stop_grad_conv1: bool = False):
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
        elif name == "vit_base":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            dim_in = model.heads.head.in_features
            model.heads = nn.Identity()
        else:
            raise NotImplementedError(f"Backbone {name} not implemented")
            
        if stop_grad_conv1:
            # Freezing the patch embedding layer (projection)
            # In torchvision ViT, this is usually model.conv_proj
            if hasattr(model, 'conv_proj'):
                 for param in model.conv_proj.parameters():
                    param.requires_grad = False
            # Some versions might cache it differently, check structure
            # But standard torchvision ViT has conv_proj.
            
        return model, dim_in
    else:
        raise ValueError(f"Unknown backbone: {name}")
