"""
U-Net architecture with ResNet-50 encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # If skip_channels is 0, it behaves like a simple upsample block
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResNetUNet(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.encoder = backbone
        
        # ResNet-50/101 channel configuration
        filters = [64, 256, 512, 1024, 2048]
        
        self.dec4 = DecoderBlock(filters[4], filters[3], 512)
        self.dec3 = DecoderBlock(512, filters[2], 256)
        self.dec2 = DecoderBlock(256, filters[1], 128)
        self.dec1 = DecoderBlock(128, filters[0], 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x0 = x 
        
        x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x) 
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        
        # Decoder
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        
        out = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.final_conv(out)
        return out

class ViTUNet(nn.Module):
    """
    Segmentation head for Vision Transformer (SETR-style Progressive Upsampling).
    """
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        self.embed_dim = backbone.hidden_dim
        
        # Progressive Upsampling Decoder (1/16 -> 1/1)
        # 4 steps of 2x upsampling
        self.dec4 = DecoderBlock(self.embed_dim, 0, 512) # 1/16 -> 1/8
        self.dec3 = DecoderBlock(512, 0, 256)           # 1/8 -> 1/4
        self.dec2 = DecoderBlock(256, 0, 128)           # 1/4 -> 1/2
        self.dec1 = DecoderBlock(128, 0, 64)            # 1/2 -> 1/1
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        
        # Forward pass through ViT
        # 1. Patch Embedding
        # Note: We manually replicate _process_input logic to ensure compatibility
        x = self.backbone._process_input(x)
        
        # 2. Transformer Encoder
        x = self.backbone.encoder(x)
        
        # 3. Reshape to Spatial
        # Remove CLS token
        x = x[:, 1:]
        # (N, L, D) -> (N, D, H/p, W/p)
        fh, fw = h // p, w // p
        x = x.permute(0, 2, 1).reshape(n, self.embed_dim, fh, fw)
        
        # Decoder (No skip connections from ViT)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        
        out = self.final_conv(x)
        return out

def get_segmentation_model(backbone, num_classes=2):
    """Factory function to return the correct U-Net based on backbone type."""
    if isinstance(backbone, VisionTransformer):
        return ViTUNet(backbone, num_classes)
    else:
        return ResNetUNet(backbone, num_classes)