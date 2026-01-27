"""
Configuration file for MoCo v3 pretraining.
Contains hyperparameters, dataset paths, and training settings.
"""

from dataclasses import dataclass
from typing import Tuple

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
    batch_size: int = 256  # Total batch size across all GPUs
    epochs: int = 100
    learning_rate: float = 0.03
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
