"""
Linear Probing for Semantic Segmentation.
Evaluates the spatial quality of representations by training a 1x1 Conv on frozen features.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import Config
from src.models.backbones import get_backbone
from src.datasets.pv03_seg import PV03SegmentationDataset

class LinearSegmentationHead(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        # backbone returns (model, dim_in)
        self.model, self.dim_in = backbone()
        self.head = nn.Conv2d(self.dim_in, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward through frozen backbone
        with torch.no_grad():
            # Need to access intermediate features before pooling
            # For ResNet, we want the output of layer4
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            
        # Linear probe (1x1 conv)
        logits = self.head(x)
        return logits

def calculate_iou(pred, target, num_classes=2):
    ious = []
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious.append(float('nan')) # Ignore if class not present
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    return ious

def main():
    parser = argparse.ArgumentParser(description='Linear Segmentation Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Pretrained Backbone
    print(f"=> Loading backbone from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Clean state dict keys (remove 'module.base_model.' prefix)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.base_model.'):
            new_state_dict[k.replace('module.base_model.', '')] = v
        elif k.startswith('base_model.'):
            new_state_dict[k.replace('base_model.', '')] = v
            
    # Initialize Model
    def backbone_fn():
        return get_backbone(config.backbone)
    
    model = LinearSegmentationHead(backbone_fn).to(device)
    
    # Load weights into backbone only
    msg = model.model.load_state_dict(new_state_dict, strict=False)
    print(f"=> Backbone loaded. Missing keys (expected for fc/head): {len(msg.missing_keys)}")

    # Freeze Backbone
    for param in model.model.parameters():
        param.requires_grad = False
    
    # 2. Data Loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Simple deterministic transform
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = PV03SegmentationDataset(config.dataset_path, transform=transform, split='train')
    val_dataset = PV03SegmentationDataset(config.dataset_path, transform=transform, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. Optimization
    optimizer = optim.SGD(model.head.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_miou = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            logits = model(images)
            # Upsample logits to match mask size
            logits = torch.nn.functional.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            
            loss = criterion(logits, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

        # Validation
        model.eval()
        total_iou = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                logits = torch.nn.functional.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
                
                batch_ious = calculate_iou(logits, masks)
                # Mean of class IoUs for this batch
                valid_ious = [iou for iou in batch_ious if not isinstance(iou, float) or not np.isnan(iou)]
                if valid_ious:
                    total_iou.append(sum(valid_ious) / len(valid_ious))

        miou = sum(total_iou) / len(total_iou) if total_iou else 0
        print(f"Epoch {epoch+1}: mIoU = {miou:.4f}")
        
        if miou > best_miou:
            best_miou = miou
            # Optional: Save best linear model

    print(f"Final Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()
