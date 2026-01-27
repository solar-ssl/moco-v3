"""
Entry point for the MoCo v3 pretraining project.
This script redirects to the main training logic in src/training/train_moco.py.
"""

import sys
from src.training.train_moco import main

if __name__ == "__main__":
    main()