"""
Dataset class for PV03 satellite imagery for SSL.
Only loads images from the original/ folder without labels.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional

class PV03SSLDataset(Dataset):
    """
    Dataset for PV03 SSL pretraining.
    Loads images from the specified directory.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            root_dir: Directory with all the images.
            transform: Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # List all image files in the directory
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {root_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
