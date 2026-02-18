"""
Checkpointing utilities for saving and loading models.
"""

import os
import torch


def save_checkpoint(state, filename='last.pth', checkpoint_dir='checkpoints'):
    """
    Saves training state to disk.

    The caller is responsible for choosing the filename:
        'last.pth'  — periodic snapshot written every save_freq epochs.
        'best.pth'  — written explicitly when avg_loss improves.

    Args:
        state:          Dict with at minimum: epoch, arch, state_dict, optimizer,
                        best_loss. state_dict must already have 'module.' stripped
                        (i.e. pass model.module.state_dict() for DDP models).
        filename:       Name of the file to write (default: 'last.pth').
        checkpoint_dir: Directory to write into. Created if it doesn't exist.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)


def load_checkpoint(model, optimizer, filename):
    """
    Loads a checkpoint into model and (optionally) optimizer.

    Always maps storage to CPU first, which is safe regardless of how many
    GPUs the checkpoint was saved on or whether the current machine has a GPU.
    The model/optimizer will be on whatever device they were already placed on
    — this function does not move them.

    Handles DDP-wrapped checkpoints transparently: if every key in the saved
    state_dict starts with 'module.', that prefix is stripped before loading
    so the weights can be loaded into a plain (non-DDP) model.

    Call this BEFORE wrapping the model with DistributedDataParallel.

    Args:
        model:     The model to load weights into. Pass None to skip.
        optimizer: The optimizer to restore state into. Pass None to skip.
        filename:  Full path to the .pth file.

    Returns:
        start_epoch (int):   Epoch to resume from (0 if file not found).
        best_loss   (float): Best loss recorded so far (inf if not in file).
    """
    if not os.path.isfile(filename):
        print(f"=> no checkpoint found at '{filename}'")
        return 0, float('inf')

    print(f"=> loading checkpoint '{filename}'")
    # map_location='cpu' is safe on any machine regardless of GPU count / index.
    # weights_only=False is required: checkpoints contain non-tensor objects
    # (rng_state dict with Python/NumPy state). Only load from trusted sources.
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)

    if model is not None:
        state_dict = checkpoint['state_dict']
        # Strip 'module.' prefix written by DistributedDataParallel
        if all(k.startswith('module.') for k in state_dict):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint.get('epoch', 0)
    best_loss   = checkpoint.get('best_loss', float('inf'))
    rng_state   = checkpoint.get('rng_state')  # None for older checkpoints

    print(f"=> loaded checkpoint '{filename}' "
          f"(epoch {start_epoch}, best_loss {best_loss:.6f})")
    return start_epoch, best_loss, rng_state
