"""
Dataset class for PV03 Semantic Segmentation.
Loads (Image, Mask) pairs for pixel-level evaluation.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional

class PV03SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None, joint_transform: Optional[Callable] = None, split: str = "train"):
        """
        Args:
            root_dir: Root directory containing 'original' and 'labels' folders.
            transform: Standard transform for image only (e.g., normalization).
            joint_transform: Function taking (image, mask) -> (image, mask). Used for consistent geometric augmentation.
            split: 'train' or 'val' (80/20 split).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.joint_transform = joint_transform
        self.img_dir = os.path.join(root_dir, "original")
        self.mask_dir = os.path.join(root_dir, "labels")
        
        # Load all images
        all_files = sorted([f for f in os.listdir(self.img_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
        
        # Simple split logic
        split_idx = int(0.8 * len(all_files))
        if split == "train":
            self.image_files = all_files[:split_idx]
        else:
            self.image_files = all_files[split_idx:]
            
        if not self.image_files:
            raise FileNotFoundError(f"No files found for split {split} in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        
        # Load Image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load Mask
        mask_path = None
        exact_path = os.path.join(self.mask_dir, img_name)
        if os.path.exists(exact_path):
            mask_path = exact_path
        else:
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.bmp', '.tif', '.jpg']:
                path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(path):
                    mask_path = path
                    break
        
        if mask_path is None:
             raise FileNotFoundError(f"No mask found for {img_name} in {self.mask_dir}")

        mask = Image.open(mask_path)

        # Apply Joint Transforms (Geometric: Crops, Flips, etc.)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        # Apply Image-only Transforms (Color, Normalization)
        if self.transform:
            image = self.transform(image)
        
        # Handle Mask format
        if not isinstance(mask, torch.Tensor):
            mask = np.array(mask)
            mask = (mask > 0).astype(np.int64) 
            mask = torch.from_numpy(mask)
        else:
            # If joint_transform returned a Tensor (e.g. standard transform wrapper), ensure it's Long
            if mask.dtype != torch.long and mask.dtype != torch.int64:
                 mask = (mask > 0).long()
            if mask.ndim == 3: # C, H, W -> Remove Channel dim if 1
                mask = mask.squeeze(0)

        return image, mask