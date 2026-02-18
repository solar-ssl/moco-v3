"""
Configuration file for MoCo v3 pretraining.
Contains hyperparameters, dataset paths, and training settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class Config:
    # Dataset settings
    dataset_path: str = "dataset/original"
    image_size: int = 224

    # Model settings
    backbone: str = "resnet50"  # Options: "resnet50", "vit_small", "vit_base"
    feature_dim: int = 256
    mlp_dim: int = 4096

    # Training settings
    batch_size: int = 64  # Total batch size (64 per GPU for 2 GPUs)
    epochs: int = 100
    temperature: float = 0.2
    # MoCo momentum (for the momentum encoder, not the SGD optimizer momentum)
    momentum: float = 0.99

    optimizer: str = "adamw"                              # Options: "adamw", "sgd"
    learning_rate: float = 1.5e-4                         # AdamW default; use 0.03 for SGD
    weight_decay: float = 0.1                             # AdamW default; use 1e-4 for SGD
    adamw_betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))
    sgd_momentum: float = 0.9                             # Only used when optimizer="sgd"
    # Maximum gradient norm for clip_grad_norm_.
    # 1.0 is the value used in the MoCo v3 paper; critical for ViT stability.
    # Set to 0.0 to disable clipping entirely (not recommended for ViT).
    clip_grad_norm: float = 1.0

    # Hardware settings
    num_workers: int = 8
    seed: Optional[int] = 42
    # When True: cudnn.deterministic=True, cudnn.benchmark=False — reproducible but slower.
    # When False: cudnn.deterministic=False, cudnn.benchmark=True — faster but non-deterministic.
    # Must be consistent with seed: if seed is set, keep deterministic=True.
    deterministic: bool = True

    # Checkpoint settings
    checkpoint_dir: str = f"checkpoints/{backbone}"
    save_freq: int = 10
