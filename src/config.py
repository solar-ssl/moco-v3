"""
Configuration file for MoCo v3 pretraining.
Contains hyperparameters, dataset paths, and training settings.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # Dataset settings
    dataset_path: str = "dataset"
    image_size: int = 512
    
    # Model settings
    backbone: str = "resnet50"  # Options: "resnet50", "vit_small", "vit_base"
    feature_dim: int = 256
    mlp_dim: int = 4096
    
    # Training settings
    batch_size: int = 32  # Total batch size (32 per GPU for 2 GPUs)
    epochs: int = 100
    learning_rate: float = 0.015
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
        return f"{bb}_bs{self.batch_size}_lr{self.learning_rate}_ep{self.epochs}"
