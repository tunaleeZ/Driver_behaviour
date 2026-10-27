"""
seed.py — Global reproducibility helpers.

Call ``set_global_seed(seed)`` at the start of any script or training run to
ensure reproducibility across Python, NumPy, PyTorch CPU/CUDA, and
DataLoader workers.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Integer seed value.  Should be stored in the run config so results
        can be reproduced exactly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic convolution algorithms (may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch seed set to %d", seed)
    except ImportError:
        logger.warning("PyTorch not installed; skipping torch seed.")

    logger.info("Global seed set to %d", seed)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init function for reproducible data loading.

    Pass as ``worker_init_fn`` argument to ``torch.utils.data.DataLoader``.

    Parameters
    ----------
    worker_id:
        Worker index (provided automatically by PyTorch DataLoader).
    """
    import numpy as _np  # local import to avoid issues if numpy not present

    worker_seed = (np.random.get_state()[1][0] + worker_id) % (2**32)
    random.seed(worker_seed)
    _np.random.seed(worker_seed)

