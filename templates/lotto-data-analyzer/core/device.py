"""Device detection utilities.

Lightweight helpers to detect GPU availability for common ML frameworks
without importing heavy libraries unless available. Safe to import from
launcher code and Streamlit pages.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List


def _run_nvidia_smi() -> List[str]:
    """Return a list of GPU lines from `nvidia-smi -L` or empty if unavailable."""
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        return lines
    except Exception:
        return []


def detect_gpus() -> Dict:
    """Detect GPU availability and which frameworks can access it.

    Returns a dict with boolean flags and a short devices list.
    This function avoids importing heavy ML libs unless already installed.
    """
    info = {
        "nvidia_smi": False,
        "nvidia_devices": [],
        "torch_cuda": False,
        "tensorflow_gpu": False,
        "cupy": False,
    }

    # nvidia-smi
    if shutil.which("nvidia-smi"):
        info["nvidia_smi"] = True
        info["nvidia_devices"] = _run_nvidia_smi()

    # PyTorch
    try:
        import torch  # type: ignore

        info["torch_cuda"] = torch.cuda.is_available()
    except Exception:
        info["torch_cuda"] = False

    # TensorFlow
    try:
        import tensorflow as tf  # type: ignore

        # tf.config.list_physical_devices('GPU') is preferred
        gpus = tf.config.list_physical_devices("GPU")
        info["tensorflow_gpu"] = len(gpus) > 0
    except Exception:
        info["tensorflow_gpu"] = False

    # CuPy
    try:
        import cupy as cp  # type: ignore

        info["cupy"] = True
    except Exception:
        info["cupy"] = False

    return info


if __name__ == "__main__":
    import json

    print(json.dumps(detect_gpus(), indent=2))
