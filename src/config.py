"""
Configuration file for MoCo v3 pretraining.
Contains hyperparameters, dataset paths, and training settings.

⚠️ HARDWARE NOTE:
This is the ORIGINAL config. For 16GB VRAM (8+8GB), use config_low_vram.py instead!
- Original: ViT-Base, batch=32 → Needs ~12-15GB per GPU (won't fit)
- Low VRAM: ResNet-50, batch=64 + grad accumulation → Fits in 8GB per GPU

Switch config in train_moco.py:
  from src.config import Config          # Original (high VRAM)
  from src.config_low_vram import Config  # Low VRAM (16GB total)
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # Dataset settings
    dataset_path: str = "dataset"
    image_size: int = 224
    
    # Model settings
    backbone: str = "vit_base"  # Options: "resnet50", "vit_small", "vit_base"
    feature_dim: int = 256
    mlp_dim: int = 4096
    
    # Training settings
    # ⚠️ WARNING: batch_size=32 with ViT-Base needs ~15GB VRAM per GPU
    batch_size: int = 32  # Total batch size (too small for pure MoCo v3)
    epochs: int = 100  # Too short for ViT (needs 300)
    learning_rate: float = 1.5e-4  # AdamW base LR (for batch=256 reference)
    warmup_epochs: int = 40
    momentum: float = 0.99
    temperature: float = 0.2
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer: str = "sgd"  # Options: "sgd", "adamw"
    
    # Hardware settings
    num_workers: int = 8
    seed: int = 42
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "moco_pv03.pth"
    save_freq: int = 10

    def get_experiment_tag(self) -> str:
        """Returns a concise tag for the current experiment configuration."""
        # Abbreviate backbone
        bb = "r50" if "resnet50" in self.backbone else self.backbone.replace("vit_", "v")
        return f"{bb}_bs{self.batch_size}_lr{self.learning_rate}_ep{self.epochs}_hybrid"
