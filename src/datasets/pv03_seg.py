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
    def __init__(self, root_dir: str, transform: Optional[Callable] = None, split: str = "train"):
        """
        Args:
            root_dir: Root directory containing 'original' and 'labels' folders.
            transform: Transform to apply to the image (must return tensor).
            split: 'train' or 'val' (80/20 split).
        """
        self.root_dir = root_dir
        self.transform = transform
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
        # Prioritize exact filename match (same name and extension)
        mask_path = None
        exact_path = os.path.join(self.mask_dir, img_name)
        if os.path.exists(exact_path):
            mask_path = exact_path
        else:
            # Fallback: try other extensions
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.bmp', '.tif', '.jpg']:
                path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(path):
                    mask_path = path
                    break
        
        if mask_path is None:
             raise FileNotFoundError(f"No mask found for {img_name} in {self.mask_dir}")

        mask = Image.open(mask_path)

        # Apply transforms
        # Note: For segmentation, we need to be careful with random transforms 
        # that change geometry (crops/flips) as they must apply to both mask and image.
        # For Linear Probing evaluation, we typically use deterministic transforms (Resize/CenterCrop).
        
        if self.transform:
            image = self.transform(image)
        
        # Process Mask
        # Resize mask to match the transformed image size if needed (simplistic approach for eval)
        # Assuming transform includes a resize to target config size (e.g., 224)
        if isinstance(image, torch.Tensor):
            target_size = image.shape[1:] # H, W
            mask = mask.resize((target_size[1], target_size[0]), resample=Image.NEAREST)
        
        mask = np.array(mask)
        # Binarize: 0 is background, 1 is solar panel (assuming >0 pixel value is panel)
        mask = (mask > 0).astype(np.int64) 
        mask = torch.from_numpy(mask)

        return image, mask
