"""
Augmentation strategies for MoCo v3 pretraining.
Implements the Two-Crop augmentation pattern suitable for satellite imagery.
"""

import random
from PIL import ImageFilter, ImageOps
from torchvision import transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
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
    Returns the MoCo v3 augmentation pipeline.
    Standard pipeline includes: ResizedCrop, ColorJitter, Grayscale, GaussianBlur, Solarize.
    """
    augmentation = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # brightness, contrast, saturation, hue
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(augmentation)
