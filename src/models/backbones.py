"""
Backbone architectures for MoCo v3.
Supports ResNet-50 and Vision Transformers.
"""

import torch.nn as nn
from torchvision import models

def get_backbone(name: str = "resnet50", pretrained: bool = False):
    """
    Returns the backbone model and its output dimension.
    
    Args:
        name: Backbone architecture name
        pretrained: Whether to use pretrained weights
        stop_grad_conv1: For ViT, freeze patch projection layer (stability trick)
                        Default True per MoCo v3 paper recommendations
    """
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # Remove the last fully connected layer
        dim_in = model.fc.in_features
        model.fc = nn.Identity()
        
        if stop_grad_conv1:
            # Freeze first conv layer for ResNet (optional stability)
            for param in model.conv1.parameters():
                param.requires_grad = False
                
        return model, dim_in
        
    elif name.startswith("vit"):
        if name == "vit_small":
            # torchvision doesn't have "small", using vit_b_16 as proxy
            # TODO: Implement actual ViT-Small architecture
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
