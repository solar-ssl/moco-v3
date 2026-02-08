"""
Backbone architectures for MoCo v3.
Supports ResNet-50 and Vision Transformers.
"""

import torch.nn as nn
from torchvision import models

def get_backbone(name: str = "resnet50", pretrained: bool = False, stop_grad_conv1: bool = True):
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
            
        if stop_grad_conv1:
            # CRITICAL STABILITY TRICK: Freeze patch projection layer
            # Per MoCo v3 paper: prevents gradient spikes in first layer
            # Torchvision ViT patch projection is in model.conv_proj
            
            patch_proj_found = False
            if hasattr(model, 'conv_proj'):
                for param in model.conv_proj.parameters():
                    param.requires_grad = False
                patch_proj_found = True
                print(f"✓ Froze ViT patch projection layer (conv_proj)")
            
            # Fallback: check encoder.conv_proj (alternative structure)
            elif hasattr(model, 'encoder') and hasattr(model.encoder, 'conv_proj'):
                for param in model.encoder.conv_proj.parameters():
                    param.requires_grad = False
                patch_proj_found = True
                print(f"✓ Froze ViT patch projection layer (encoder.conv_proj)")
                
            if not patch_proj_found:
                raise RuntimeError(
                    f"Cannot find patch projection layer in {name}. "
                    f"Model structure: {list(model._modules.keys())}. "
                    f"stop_grad_conv1=True requires identifiable conv_proj layer."
                )
            
        return model, dim_in
    else:
        raise ValueError(f"Unknown backbone: {name}")
