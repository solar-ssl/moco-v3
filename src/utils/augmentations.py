"""
Augmentation strategies for MoCo v3 pretraining.
Implements the Two-Crop augmentation pattern suitable for satellite imagery.
"""

import random
from PIL import ImageFilter, ImageOps
from torchvision import transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        q = self.base_transform1(x)
        k = self.base_transform2(x)
        return [q, k]

class GaussianBlur:
    """Gaussian blur augmentation from SimCLR/MoCo v2."""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

class Solarize:
    """Solarize augmentation from BYOL/MoCo v3."""
    def __call__(self, x):
        return ImageOps.solarize(x)

def get_moco_v3_augmentations(image_size: int = 224):
    """
    Returns two MoCo v3 augmentation pipelines adapted for satellite imagery.
    Pipeline 1: Strong Jitter + Blur
    Pipeline 2: Strong Jitter + Solarize + Rare Blur
    """
    # Base augmentations shared by both
    base_aug = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5), # Crucial for satellite imagery
    ]
    
    # Stronger color jitter
    color_jitter = transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    ], p=0.8)
    
    aug1 = base_aug + [
        color_jitter,
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    aug2 = base_aug + [
        color_jitter,
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(aug1), transforms.Compose(aug2)
