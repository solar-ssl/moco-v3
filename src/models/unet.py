"""
U-Net architecture with ResNet-50 encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            # Handle slight shape mismatches if any
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
        # Stem (64) -> L1 (256) -> L2 (512) -> L3 (1024) -> L4 (2048)
        filters = [64, 256, 512, 1024, 2048]
        
        # Decoder Blocks
        # Input: 2048, Skip: 1024 -> Out: 512
        self.dec4 = DecoderBlock(filters[4], filters[3], 512)
        # Input: 512, Skip: 512 -> Out: 256
        self.dec3 = DecoderBlock(512, filters[2], 256)
        # Input: 256, Skip: 256 -> Out: 128
        self.dec2 = DecoderBlock(256, filters[1], 128)
        # Input: 128, Skip: 64 -> Out: 64
        self.dec1 = DecoderBlock(128, filters[0], 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder (ResNet)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x0 = x # Stride 2 (64 ch)
        
        x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x) # Stride 4 (256 ch)
        x2 = self.encoder.layer2(x1) # Stride 8 (512 ch)
        x3 = self.encoder.layer3(x2) # Stride 16 (1024 ch)
        x4 = self.encoder.layer4(x3) # Stride 32 (2048 ch)
        
        # Decoder
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        
        # Final Upsample to original resolution (Stride 2 -> 1)
        out = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.final_conv(out)
        
        return out
