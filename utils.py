"""Shared utilities for RoBERTa Diffusion project.

Common functions used across multiple scripts to avoid code duplication.
"""

import torch


def get_device() -> torch.device:
    """Select the best available device (MPS > CUDA > CPU).

    Returns:
        torch.device: Best available device
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("[INFO] Using MPS (Apple silicon) backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[INFO] Using CUDA backend")
        return torch.device("cuda")
    else:
        print("[INFO] Using CPU backend")
        return torch.device("cpu")
