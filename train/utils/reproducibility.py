from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int):
    """Set global RNG seeds for Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Worker initialization function for DataLoader to ensure reproducibility.

    This function should be passed as `worker_init_fn` to DataLoader when
    num_workers > 0. It sets the random seed for each worker based on the
    base seed and worker_id to ensure different workers have different but
    deterministic random states.

    Args:
        worker_id: The ID of the worker process (provided by DataLoader)
    """
    # Get the base seed from torch's initial seed (set by set_global_seed)
    base_seed = torch.initial_seed()
    # Create a unique seed for each worker
    worker_seed = base_seed + worker_id
    # Set seeds for all random number generators in the worker
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))  # numpy seed must be within 32-bit
    torch.manual_seed(worker_seed)
