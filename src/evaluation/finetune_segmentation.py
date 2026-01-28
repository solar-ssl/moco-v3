"""
Fine-tuning script for Semantic Segmentation using U-Net.
Benchmarks the downstream performance of the pretrained backbone.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import Config
from src.models.backbones import get_backbone
from src.models.unet import ResNetUNet
from src.datasets.pv03_seg import PV03SegmentationDataset

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
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    return ious

def main():
    parser = argparse.ArgumentParser(description='U-Net Fine-tuning')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to MoCo checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save-dir', type=str, default='checkpoints_finetune')
    args = parser.parse_args()

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 1. Initialize Backbone
    # If --pretrained is set, get_backbone loads ImageNet weights
    def backbone_fn():
        return get_backbone(config.backbone, pretrained=args.pretrained)

    backbone_model, _ = backbone_fn()

    # 2. Load MoCo Weights if provided (Overwrites ImageNet/Random)
    if args.checkpoint:
        print(f"=> Loading MoCo weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Clean keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.base_model.'):
                new_state_dict[k.replace('module.base_model.', '')] = v
            elif k.startswith('base_model.'):
                new_state_dict[k.replace('base_model.', '')] = v
        
        # Load into backbone
        msg = backbone_model.load_state_dict(new_state_dict, strict=False)
        print(f"=> Loaded. Missing keys: {len(msg.missing_keys)}")
    
    # 3. Build U-Net
    model = ResNetUNet(backbone_model, num_classes=2).to(device)

    # 4. Data Loading
    # Use standard norms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Training transforms (can add Augmentations here for better performance)
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = PV03SegmentationDataset(config.dataset_path, transform=train_transform, split='train')
    val_dataset = PV03SegmentationDataset(config.dataset_path, transform=val_transform, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 5. Optimization
    # Optimize all parameters (Fine-tuning)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            output = model(images)
            
            # CrossEntropy expects LongTensor masks
            loss = criterion(output, masks.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'lr': optimizer.param_groups[0]['lr']})
        
        scheduler.step()

        # Validation
        model.eval()
        total_ious = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(device), masks.to(device)
                output = model(images)
                
                batch_ious = calculate_iou(output, masks)
                valid_ious = [iou for iou in batch_ious if not np.isnan(iou)]
                if valid_ious:
                    total_ious.append(np.mean(valid_ious))
        
        miou = np.mean(total_ious) if total_ious else 0
        print(f"Epoch {epoch+1}: mIoU = {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_unet.pth'))
            print("=> Saved best model")

    print(f"Final Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()
