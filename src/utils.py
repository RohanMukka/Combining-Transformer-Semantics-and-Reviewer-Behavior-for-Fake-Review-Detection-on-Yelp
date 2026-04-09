"""
Utility Functions for ReviewGuard
==================================

Provides common helpers used across the pipeline:
  - Reproducibility (set_seed)
  - Device selection (get_device)
  - Logging setup (setup_logging)
  - Config loading (load_config)
  - Directory creation (ensure_dirs)
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import yaml


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic ops (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logging.getLogger(__name__).debug(f"All random seeds set to {seed}.")


# ─── Device Selection ─────────────────────────────────────────────────────────

def get_device(device: str = "auto") -> "torch.device":
    """Return the best available PyTorch device.

    Args:
        device: One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
                ``"auto"`` picks CUDA > MPS > CPU in that order.

    Returns:
        A :class:`torch.device` object.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            selected = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected = torch.device("mps")
        else:
            selected = torch.device("cpu")
    else:
        selected = torch.device(device)

    logging.getLogger(__name__).info(f"Using device: {selected}")
    return selected


# ─── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    """Configure the root logger with a console (and optionally file) handler.

    Args:
        level: Logging level string, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        log_file: Optional path to a log file. If provided, logs are written both
                  to the console and to the file.
        fmt: Log message format string.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        handlers=handlers,
    )
    logging.getLogger(__name__).debug("Logging configured.")


# ─── Config Loading ───────────────────────────────────────────────────────────

def load_config(config_path: Union[str, Path] = "configs/default_config.yaml") -> Dict:
    """Load a YAML configuration file into a nested dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    logging.getLogger(__name__).info(f"Config loaded from {config_path}")
    return config


# ─── Directory Utilities ──────────────────────────────────────────────────────

def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """Create directories (including parents) if they do not already exist.

    Args:
        *dirs: One or more directory paths to create.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    """Return the absolute path to the project root directory.

    Assumes this file lives at ``<project_root>/src/utils.py``.

    Returns:
        :class:`pathlib.Path` pointing to the project root.
    """
    return Path(__file__).resolve().parent.parent


# ─── Metric Formatting ────────────────────────────────────────────────────────

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format a dictionary of metric values into a human-readable string.

    Args:
        metrics: Mapping from metric name to float value.
        prefix: Optional string prepended to each metric name.

    Returns:
        A formatted string, e.g. ``"auc_roc=0.8901  macro_f1=0.8312"``.
    """
    parts = [f"{prefix}{k}={v:.4f}" for k, v in sorted(metrics.items())]
    return "  ".join(parts)


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging("DEBUG")
    logger = logging.getLogger(__name__)

    logger.info("ReviewGuard Utilities — smoke test")
    set_seed(42)

    try:
        device = get_device("auto")
        logger.info(f"Device selected: {device}")
    except ImportError:
        logger.warning("PyTorch not installed; skipping device test.")

    cfg = load_config()
    logger.info(f"Loaded config keys: {list(cfg.keys())}")

    ensure_dirs("data/raw", "data/processed", "data/features")
    logger.info("Directories ensured.")
