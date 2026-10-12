"""Device selection, hardware description, memory and RNG-state helpers."""
from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def hardware_info(device: torch.device) -> dict:
    info = dict(platform=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, device=str(device),
                cuda_available=torch.cuda.is_available(),
                torch_num_threads=torch.get_num_threads(),
                logical_cpus=os.cpu_count())
    proc = platform.processor()
    try:  # nicer CPU name on Windows
        import subprocess
        if os.name == "nt":
            proc = subprocess.run(["powershell", "-NoProfile", "-Command",
                                   "(Get-CimInstance Win32_Processor).Name"],
                                  capture_output=True, text=True, timeout=20).stdout.strip() or proc
    except Exception:
        pass
    info["cpu"] = proc
    if psutil is not None:
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    if psutil is None:
        return None
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)  # Windows
    if peak is None:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: kB
        except Exception:
            peak = mi.rss
    return peak / 2 ** 20


def get_rng_states() -> dict:
    st = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_states(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
