"""Central configuration and utilities for RoBERTa Diffusion project.

All configuration classes and shared utilities in one place.
"""

import torch


class Config:
    """Unified configuration for RoBERTa Diffusion project."""

    # Model config
    MODEL_NAME: str = "roberta-base"
    PREFIX_LEN: int = 32
    MAX_LEN: int = 256
    CONFIDENCE_THRESHOLD: float = 0.9
    TEMPERATURE: float = 0.8

    # GPT-2 specific
    GPT_MODEL_NAME: str = "gpt2"
    TOP_K: int = 50
    TOP_P: float = 0.95

    # Training
    BATCH_SIZE: int = 16
    OUTPUT_DIR: str = "weights"
    MAX_STEPS: int = 1000
    SAVE_STEPS: int = 500
    LOGGING_STEPS: int = 50
    SAVE_TOTAL_LIMIT: int = 1

    # Animation
    ANIMATION_FPS: int = 30


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
