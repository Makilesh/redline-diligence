"""
Whether something else was computing on the GPU while a benchmark timed itself.

Measured on the development laptop (round 12): another process generating text on the
GPU multiplied the golden set's p95 latencies by 3.8 to 59 (production 229 -> 2,013 ms) and its wall
time by 2.4, while one that only held memory — up to 9.2 GB of the 12, idle — changed no
timing by more than 5%, because the driver pages an idle process's memory out. Quality was
identical in all thirteen runs. So a timing can be trusted only when the GPU was otherwise
idle, and the signal for that is utilisation, not memory in use.

The harnesses sample utilisation just before they load their models and again when they
finish, and every results file records both, so a timing taken on a busy GPU says so.
Another process that starts and stops mid-run is not seen.

Best effort: no nvidia-smi, no NVIDIA GPU, or any error reading it gives
{"available": False}; a benchmark never fails because of this module.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable

# Mean utilisation, in percent, at or above which the GPU counts as busy. An idle desktop
# GPU reads 0-3% (a browser repainting); the generating model above read 40-100%.
BUSY_UTILISATION_PCT = 10.0

_FIELDS = "name,memory.used,memory.total,utilization.gpu"


def _nvidia_smi(fields: str) -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = subprocess.run(
        [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.strip().splitlines()[0]


def parse_gpu_line(line: str) -> dict:
    """One `name, memory.used, memory.total, utilization.gpu` line (MiB, MiB, %)."""
    name, used, total, util = (part.strip() for part in line.rsplit(",", 3))
    return {
        "name": name,
        "memory_used_mb": float(used),
        "memory_total_mb": float(total),
        "utilization_pct": float(util),
    }


def sample(seconds: float = 2.0, interval: float = 0.5,
           query: Callable[[str], str | None] = _nvidia_smi) -> dict:
    """
    Utilisation and memory of the first GPU over a short window.

    Returns:
        {"available": False}, or {"available": True, "name", "memory_used_mb" (max),
        "memory_total_mb", "utilization_pct_mean", "utilization_pct_max", "samples"}.
    """
    try:
        readings = []
        deadline = time.monotonic() + seconds
        while True:
            line = query(_FIELDS)
            if line is None:
                return {"available": False}
            readings.append(parse_gpu_line(line))
            if time.monotonic() >= deadline:
                break
            time.sleep(interval)
        util = [r["utilization_pct"] for r in readings]
        return {
            "available": True,
            "name": readings[-1]["name"],
            "memory_used_mb": max(r["memory_used_mb"] for r in readings),
            "memory_total_mb": readings[-1]["memory_total_mb"],
            "utilization_pct_mean": round(sum(util) / len(util), 1),
            "utilization_pct_max": max(util),
            "samples": len(readings),
        }
    except Exception:
        return {"available": False}


def sample_if(device: str, **kwargs) -> dict:
    """sample() when the benchmark runs on CUDA ("cuda", "cuda (<name>)"), else unavailable."""
    return sample(**kwargs) if str(device).startswith("cuda") else {"available": False}


def sample_after(before: dict, settle: float = 2.0, **kwargs) -> dict:
    """
    The closing sample, taken only when the opening one could read the GPU.

    nvidia-smi's utilisation is the share of its last sample window (up to a second) in
    which any kernel ran, so a reading taken right after the benchmark's last batch counts
    the benchmark itself: the first golden run with this module read 85% that way, with
    nothing else on the GPU. Waiting `settle` seconds first leaves only other processes.
    """
    if not (before or {}).get("available"):
        return {"available": False}
    time.sleep(settle)
    return sample(**kwargs)


def busy(snapshot: dict | None) -> bool:
    """True when a snapshot shows the GPU computing for someone else."""
    return bool(snapshot and snapshot.get("available")) and \
        snapshot.get("utilization_pct_mean", 0.0) >= BUSY_UTILISATION_PCT


def record(before: dict, after: dict) -> dict:
    """The `gpu` entry of a results file."""
    return {"before": before, "after": after, "busy": busy(before) or busy(after)}


def note(gpu: dict | None) -> str:
    """A warning line for a results summary, or "" when the GPU was idle or unread."""
    if not gpu or not gpu.get("busy"):
        return ""
    seen = [f"{gpu[when]['utilization_pct_mean']:.0f}% {when} this run (peak {gpu[when]['utilization_pct_max']:.0f}%)"
            for when in ("before", "after") if busy(gpu.get(when))]
    return ("**Timings not comparable:** another process was computing on the GPU, at "
            f"{' and '.join(seen)}. Quality figures are unaffected.")
