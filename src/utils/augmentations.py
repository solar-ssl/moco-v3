"""
Augmentation strategies for MoCo v3 pretraining.

Implements the ASYMMETRIC two-view augmentation strategy from the MoCo v3 paper
(Chen et al., 2021 - https://arxiv.org/abs/2104.02057), Table 7.

Key asymmetry between the two views:
    View 1 (query)  — GaussianBlur p=1.0,  Solarize p=0.0
    View 2 (key)    — GaussianBlur p=0.1,  Solarize p=0.2

Additional satellite-imagery augmentations added on top of the base pipeline:
    - RandomVerticalFlip  : top-down imagery has no canonical vertical orientation
    - BICUBIC interpolation on RandomResizedCrop : better quality at sub-pixel scales
"""

import random
from PIL import ImageFilter, ImageOps
from torchvision import transforms

class GaussianBlur:
    """
    Gaussian blur with a randomly sampled radius in [sigma_min, sigma_max].
    Identical to the implementation used in SimCLR, MoCo v2 and MoCo v3.
    """
    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

class Solarize:
    """
    Inverts all pixel values above a threshold (128).
    Used in BYOL and MoCo v3, applied only to view 2.
    """
    def __call__(self, x):
        return ImageOps.solarize(x)

class TwoCropsTransform:
    """
    Applies two *different* augmentation pipelines to the same PIL image and
    returns [view1, view2].  The transforms should be built with
    get_moco_v3_augmentations(), which sets the paper-correct probabilities.
    """
    def __init__(self, transform_q, transform_k):
        self.transform_q = transform_q   # query  pipeline
        self.transform_k = transform_k   # key pipeline

    def __call__(self, x):
        return [self.transform_q(x), self.transform_k(x)]

def _base_pipeline(image_size: int) -> list:
    """
    Augmentations shared by both views:
      crop → color jitter → grayscale → flip (h+v)
    """
    return [
        # BICUBIC gives sharper crops than bilinear at small scales
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        # Satellite imagery has no canonical up/down direction
        transforms.RandomVerticalFlip(p=0.5),
    ]

def _normalise() -> list:
    """Standard ImageNet normalisation applied after ToTensor."""
    return [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

def _build_view1(image_size: int) -> transforms.Compose:
    """
    Query view (view 1) pipeline — MoCo v3 paper Table 7, left column.

    Blur:     p=1.0  (always applied)
    Solarize: p=0.0  (never applied)

    The high-probability blur makes this a 'harder' view where fine textures
    are destroyed, forcing the model to rely on structure and semantics.
    """
    return transforms.Compose(
        _base_pipeline(image_size) + [
            transforms.RandomApply([GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=1.0),
            # No solarize for view 1
        ] + _normalise()
    )

def _build_view2(image_size: int) -> transforms.Compose:
    """
    Key view (view 2) pipeline — MoCo v3 paper Table 7, right column.

    Blur:     p=0.1  (rarely applied)
    Solarize: p=0.2  (occasionally inverts pixel intensities)

    Keeping view 2 sharper (less blur) while sometimes solarising creates
    a complementary difficulty profile to view 1.
    """
    return transforms.Compose(
        _base_pipeline(image_size) + [
            transforms.RandomApply([GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
        ] + _normalise()
    )

def get_moco_v3_augmentations(image_size: int = 224) -> TwoCropsTransform:
    """
    Returns a TwoCropsTransform that produces the asymmetric view pair used in
    the MoCo v3 paper.  Pass the result directly as the `transform` argument
    of PV03SSLDataset — no further wrapping is needed.

    Args:
        image_size: spatial size for RandomResizedCrop (default: 224).

    Returns:
        TwoCropsTransform(view1_pipeline, view2_pipeline)
    """
    return TwoCropsTransform(
        transform_q=_build_view1(image_size),
        transform_k=_build_view2(image_size),
    )
