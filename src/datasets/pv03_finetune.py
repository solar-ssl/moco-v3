"""
Supervised dataset for fine-tuning the U-Net segmentation model on PV03
satellite imagery.

Directory layout expected
─────────────────────────
  Training (BMP images and masks with matching filenames):
      dataset/original/*.bmp   — RGB satellite patches
      dataset/labels/*.bmp     — binary segmentation masks

  Test (PNG images and masks with matching filenames):
      dataset/test_local/original/*.png
      dataset/test_local/labels/*.png

Label convention
────────────────
  Masks are expected to be grayscale images where:
      0       → background
      non-zero → solar panel (normalised to 1.0 by this dataset)

  Any pixel value above 0 is treated as foreground.  This handles both
  0/255 BMP masks and already-binary 0/1 PNG masks transparently.

Usage
─────
  from src.datasets.pv03_finetune import PV03FinetuneDataset, get_finetune_transforms

  train_tf, val_tf = get_finetune_transforms(image_size=224)

  full_dataset = PV03FinetuneDataset(
      image_dir = "dataset/original",
      label_dir = "dataset/labels",
      transform = train_tf,
  )
  # 80/20 reproducible split
  train_ds, val_ds = full_dataset.split(val_fraction=0.2, seed=42)
  val_ds.transform = val_tf   # swap to val transforms after split
"""

import os
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# ─── Augmentations ────────────────────────────────────────────────────────────

def get_finetune_transforms(
    image_size: int = 224,
) -> Tuple[Callable, Callable]:
    """
    Returns (train_transform, val_transform).

    Training augmentations
    ───────────────────────
    Both the image and mask receive identical spatial augmentations so that
    pixels remain aligned.  The mask then gets only ToTensor (no normalisation
    — it is a 0/1 binary map).

    Augmentations used during training:
        • RandomResizedCrop  (scale 0.5–1.0, BICUBIC for image, NEAREST for mask)
        • RandomHorizontalFlip
        • RandomVerticalFlip
        • ColorJitter on image only (brightness, contrast, saturation, hue)

    Validation augmentations
    ─────────────────────────
    Resize + CenterCrop only (deterministic).
    """
    _resize = int(image_size * 1.15)
    _mean   = [0.485, 0.456, 0.406]
    _std    = [0.229, 0.224, 0.225]
    # Instantiate once so the same distribution object is reused every call
    _jitter = transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
    )

    def _train_tf(image: Image.Image, mask: Image.Image):
        """
        All geometric operations are applied to image and mask with identical
        parameters to preserve pixel alignment.  Colour jitter is image-only.

        Pipeline: Resize → synchronized RandomCrop → random flips → jitter
                  → ToTensor + Normalize (image) / ToTensor only (mask)
        """
        # 1. Resize both (slightly oversized for the random crop to bite into)
        image = TF.resize(image, _resize, interpolation=transforms.InterpolationMode.BICUBIC)
        mask  = TF.resize(mask,  _resize, interpolation=transforms.InterpolationMode.NEAREST)

        # 2. Synchronized RandomCrop — compute (i, j, h, w) once, apply to both
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(image_size, image_size)
        )
        image = TF.crop(image, i, j, h, w)
        mask  = TF.crop(mask,  i, j, h, w)

        # 3. Synchronized random flips
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # 4. Colour jitter on image only (mask must not be colour-jittered)
        image = _jitter(image)

        # 5. Convert to tensor and normalise
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=_mean, std=_std)
        mask  = TF.to_tensor(mask)
        return image, mask

    def _val_tf(image: Image.Image, mask: Image.Image):
        """Deterministic resize + centre crop only."""
        image = TF.resize(image, _resize, interpolation=transforms.InterpolationMode.BICUBIC)
        mask  = TF.resize(mask,  _resize, interpolation=transforms.InterpolationMode.NEAREST)
        image = TF.center_crop(image, image_size)
        mask  = TF.center_crop(mask,  image_size)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=_mean, std=_std)
        mask  = TF.to_tensor(mask)
        return image, mask

    return _train_tf, _val_tf


# ─── Dataset ──────────────────────────────────────────────────────────────────

# Image file extensions the dataset will search for
_IMAGE_EXTS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _find_pairs(image_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    """
    Discover (image_path, label_path) pairs by matching filenames (stem only,
    ignoring extension).  Both directories are searched for any supported
    image extension so that BMP↔PNG mismatches are handled transparently.

    Returns a sorted list of pairs to guarantee a deterministic order.
    """
    # Build stem → full-path mapping for labels
    label_map = {}
    for fname in os.listdir(label_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in _IMAGE_EXTS:
            label_map[stem] = os.path.join(label_dir, fname)

    pairs = []
    for fname in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in _IMAGE_EXTS:
            continue
        if stem in label_map:
            pairs.append((
                os.path.join(image_dir, fname),
                label_map[stem],
            ))

    if not pairs:
        raise FileNotFoundError(
            f"No matching (image, label) pairs found.\n"
            f"  image_dir: {image_dir}\n"
            f"  label_dir: {label_dir}\n"
            f"Make sure both directories contain files with matching stems."
        )
    return pairs


class PV03FinetuneDataset(Dataset):
    """
    Paired (image, mask) dataset for supervised fine-tuning.

    Args:
        image_dir:  Directory containing RGB image files.
        label_dir:  Directory containing corresponding binary mask files.
                    Masks are matched to images by filename stem (ignoring
                    extension), so BMP images can pair with PNG masks.
        transform:  A callable (image, mask) → (tensor, tensor).
                    Use ``get_finetune_transforms()`` to obtain train/val
                    transforms.  If None, returns raw PIL images.

    Notes:
        • Masks are binarised: any pixel > 0 becomes 1.0.
        • ``split(val_fraction, seed)`` returns two datasets that share the
          same ``pairs`` list — the split is reproducible across runs.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.pairs     = _find_pairs(image_dir, label_dir)

    # ── internal constructor used by split() ──────────────────────────────────
    @classmethod
    def _from_pairs(
        cls,
        pairs: List[Tuple[str, str]],
        transform: Optional[Callable],
    ) -> "PV03FinetuneDataset":
        obj           = cls.__new__(cls)
        obj.pairs     = pairs
        obj.transform = transform
        return obj

    # ── public API ────────────────────────────────────────────────────────────

    def split(
        self,
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> Tuple["PV03FinetuneDataset", "PV03FinetuneDataset"]:
        """
        Split this dataset into a training set and a validation set.

        The split is reproducible (uses ``seed``) and stratified over the
        sorted file list so that both halves cover a similar range of image
        names.

        The returned validation dataset inherits the *same* transform as
        the parent; call ``val_ds.transform = val_tf`` afterwards to swap
        it for the deterministic val transform.

        Returns:
            (train_dataset, val_dataset)
        """
        rng = random.Random(seed)
        indices = list(range(len(self.pairs)))
        rng.shuffle(indices)

        n_val    = max(1, int(len(indices) * val_fraction))
        val_idx  = set(indices[:n_val])
        train_idx = [i for i in range(len(self.pairs)) if i not in val_idx]
        val_idx   = sorted(val_idx)

        train_pairs = [self.pairs[i] for i in train_idx]
        val_pairs   = [self.pairs[i] for i in val_idx]

        return (
            PV03FinetuneDataset._from_pairs(train_pairs, self.transform),
            PV03FinetuneDataset._from_pairs(val_pairs,   self.transform),
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.pairs[idx]

        max_retries = 5
        for attempt in range(max_retries):
            current = (idx + attempt) % len(self.pairs)
            img_path, lbl_path = self.pairs[current]
            try:
                image = Image.open(img_path).convert('RGB')
                mask  = Image.open(lbl_path).convert('L')    # grayscale
                break
            except Exception as exc:
                print(
                    f"[PV03FinetuneDataset] Warning: could not load "
                    f"'{img_path}': {exc}. Trying next sample "
                    f"(attempt {attempt + 1}/{max_retries})."
                )
        else:
            raise RuntimeError(
                f"Failed to load {max_retries} consecutive samples starting "
                f"at index {idx}. Check your dataset for corrupt files."
            )

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # Return PIL images if no transform is set
            return image, mask  # type: ignore[return-value]

        # Binarise mask: non-zero → 1.0, zero → 0.0.
        # threshold=1e-3 safely handles both:
        #   • 0/255 uint8 masks → ToTensor gives {0.0, 1.0}        → > 1e-3 ✓
        #   • 0/1   uint8 masks → ToTensor gives {0.0, 0.004} (1÷255) → > 1e-3 ✓
        mask = (mask > 1e-3).float()

        return image, mask
