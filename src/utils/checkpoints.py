"""
Checkpointing utilities for saving and loading models.
"""

import os
import torch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='checkpoints'):
    """Saves the training state."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, filename):
    """Loads a training state."""
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return start_epoch
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return 0
