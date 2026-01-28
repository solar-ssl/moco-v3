"""
Fine-tuning script for Semantic Segmentation using U-Net.
Implements Research-Grade Tiled Training and Sliding Window Inference.
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from src.config import Config
from src.models.backbones import get_backbone
from src.models.unet import ResNetUNet
from src.datasets.pv03_seg import PV03SegmentationDataset

# --- Joint Transforms ---
class JointRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        # image: PIL Image, mask: PIL Image
        # Random Crop
        w, h = image.size
        th, tw = self.size, self.size
        if w == tw and h == th:
            return image, mask
        
        if w < tw or h < th:
            # Resize if smaller (edge case)
            return transforms.Resize((th, tw))(image), transforms.Resize((th, tw), interpolation=Image.NEAREST)(mask)

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        return image.crop((j, i, j + tw, i + th)), mask.crop((j, i, j + tw, i + th))

class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

class ComposeJoint:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

# --- Inference Helper ---
def sliding_window_inference(model, image, num_classes, crop_size=224, stride=112, device='cuda'):
    """
    Performs sliding window inference on a single image.
    image: Tensor (C, H, W)
    Returns: Tensor (H, W) class predictions
    """
    model.eval()
    _, H, W = image.shape
    
    # Probability map accumulator
    probs = torch.zeros((num_classes, H, W), device=device)
    count = torch.zeros((1, H, W), device=device)
    
    # Generate grid
    rows = int(np.ceil((H - crop_size) / stride)) + 1
    cols = int(np.ceil((W - crop_size) / stride)) + 1
    
    with torch.no_grad():
        for r in range(rows):
            for c in range(cols):
                y1 = int(r * stride)
                x1 = int(c * stride)
                y2 = min(y1 + crop_size, H)
                x2 = min(x1 + crop_size, W)
                y1 = max(y2 - crop_size, 0)
                x1 = max(x2 - crop_size, 0)
                
                crop = image[:, y1:y2, x1:x2].unsqueeze(0).to(device) # (1, C, H, W)
                output = model(crop)
                
                probs[:, y1:y2, x1:x2] += output.squeeze(0)
                count[:, y1:y2, x1:x2] += 1

    probs /= count
    pred = torch.argmax(probs, dim=0) # (H, W)
    return pred

def calculate_iou(pred, target, num_classes=2):
    ious = []
    # pred: (H, W), target: (H, W)
    pred = pred.view(-1)
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
    parser = argparse.ArgumentParser(description='Research-Grade U-Net Fine-tuning')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to MoCo checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save-dir', type=str, default='checkpoints_finetune')
    parser.add_argument('--crop-size', type=int, default=224, help='Training crop size')
    parser.add_argument('--val-subset', type=int, default=50, help='Number of validation images to check during training (speed up)')
    args = parser.parse_args()

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 1. Initialize Backbone
    def backbone_fn():
        return get_backbone(config.backbone, pretrained=args.pretrained)

    backbone_model, _ = backbone_fn()

    # 2. Load MoCo Weights
    if args.checkpoint:
        print(f"=> Loading MoCo weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.base_model.'):
                new_state_dict[k.replace('module.base_model.', '')] = v
            elif k.startswith('base_model.'):
                new_state_dict[k.replace('base_model.', '')] = v
        backbone_model.load_state_dict(new_state_dict, strict=False)
    
    model = ResNetUNet(backbone_model, num_classes=2).to(device)

    # 3. Data Loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Training: Random Crop 224x224 + Flips
    train_joint = ComposeJoint([
        JointRandomCrop(args.crop_size),
        JointRandomHorizontalFlip()
    ])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation: No resize, full resolution!
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = PV03SegmentationDataset(config.dataset_path, 
                                            transform=train_transform, 
                                            joint_transform=train_joint, 
                                            split='train')
    val_dataset = PV03SegmentationDataset(config.dataset_path, 
                                          transform=val_transform, 
                                          joint_transform=None, # No cropping for validation
                                          split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # Create subset for intermediate validation
    if args.val_subset > 0 and args.val_subset < len(val_dataset):
        indices = list(range(args.val_subset))
        val_subset_dataset = Subset(val_dataset, indices)
    else:
        val_subset_dataset = val_dataset

    # Loaders for subset (fast) and full (final)
    val_loader_fast = DataLoader(val_subset_dataset, batch_size=1, shuffle=False, num_workers=4)
    val_loader_full = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 4. Optimization
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_miou = 0.0

    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            output = model(images)
            loss = criterion(output, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'lr': optimizer.param_groups[0]['lr']})
        
        scheduler.step()

        # Validation Phase
        # Use full set on last epoch, subset otherwise
        is_last_epoch = (epoch == args.epochs - 1)
        current_val_loader = val_loader_full if is_last_epoch else val_loader_fast
        loader_desc = "Validating (Full)" if is_last_epoch else "Validating (Subset)"
        
        model.eval()
        total_ious = []
        with torch.no_grad():
            for images, masks in tqdm(current_val_loader, desc=loader_desc):
                img = images[0] 
                mask = masks[0].to(device)
                
                # Run sliding window
                pred = sliding_window_inference(model, img, num_classes=2, crop_size=args.crop_size, device=device)
                
                iou = calculate_iou(pred, mask)
                valid_ious = [x for x in iou if not np.isnan(x)]
                if valid_ious:
                    total_ious.append(np.mean(valid_ious))
        
        miou = np.mean(total_ious) if total_ious else 0
        print(f"Epoch {epoch+1}: mIoU = {miou:.4f}")

        # Save best model (track performance on subset for checkpoints)
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_unet.pth'))
            print("=> Saved best model")

    print(f"Final Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()
