"""
Augmentation strategies for MoCo v3 pretraining.
Implements the Two-Crop augmentation pattern suitable for satellite imagery.
"""

import random
from PIL import ImageFilter, ImageOps
import torch
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

class GaussianNoise:
    """Gaussian noise augmentation for satellite imagery."""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        # We assume image is already a tensor or we convert it
        if not isinstance(img, torch.Tensor):
            t_img = transforms.ToTensor()(img)
        else:
            t_img = img
            
        noise = torch.randn(t_img.size()) * self.std + self.mean
        noisy_img = t_img + noise
        
        # Clip to valid range [0, 1]
        noisy_img = torch.clamp(noisy_img, 0., 1.)
        
        # If input was PIL, convert back (though usually we apply this after ToTensor)
        if not isinstance(img, torch.Tensor):
            return transforms.ToPILImage()(noisy_img)
        return noisy_img

def get_moco_v3_augmentations(image_size: int = 224):
    """
    Returns the MoCo v3 augmentation pipeline.
    Standard pipeline includes: ResizedCrop, ColorJitter, Grayscale, GaussianBlur, Solarize.
    Enhanced with RandomRotation and GaussianNoise for satellite imagery.
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
        transforms.RandomRotation(90), # 90 degree rotations for man-made structures
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([GaussianNoise(std=0.05)], p=0.5) # Apply noise after normalization
    ]
    return transforms.Compose(augmentation)
