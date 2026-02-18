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
        
        # Sort to guarantee identical ordering across filesystems and OSes.
        # os.listdir() returns files in arbitrary filesystem-dependent order
        # which differs between ext4/NTFS/S3 and can vary between runs.
        self.image_files = sorted(
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
        )
        
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {root_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        max_retries = 5
        for attempt in range(max_retries):
            current_idx = (idx + attempt) % len(self.image_files)
            img_name = os.path.join(self.root_dir, self.image_files[current_idx])
            try:
                image = Image.open(img_name).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image
            except Exception as exc:
                print(
                    f"[PV03SSLDataset] Warning: could not load '{img_name}': {exc}. "
                    f"Trying next image (attempt {attempt + 1}/{max_retries})."
                )
        raise RuntimeError(
            f"Failed to load {max_retries} consecutive images starting at index {idx}. "
            f"Check your dataset for corrupt files."
        )
