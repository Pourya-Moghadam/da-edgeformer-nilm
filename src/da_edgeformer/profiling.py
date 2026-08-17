from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass

import numpy as np
import torch

from da_edgeformer.models.edgeformer import DAEdgeFormer


@dataclass
class InferenceProfile:
    p50_ms: float
    p95_ms: float
    rss_mb: float
    samples_ms: list[float]
    parameters_total: int
    parameters_trainable: int
    torch_threads: int


def profile_inference(
    model: DAEdgeFormer,
    window_size: int,
    device: torch.device,
    warmup: int = 100,
    iterations: int = 1000,
    threads: int = 1,
) -> InferenceProfile:
    torch.set_num_threads(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    model = model.to(device).eval()
    sample = torch.zeros(1, window_size, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        durations = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            model(sample)
            durations.append((time.perf_counter_ns() - started) / 1e6)
    counts = model.parameter_counts()
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return InferenceProfile(
        p50_ms=float(np.percentile(durations, 50)),
        p95_ms=float(np.percentile(durations, 95)),
        rss_mb=float(rss_kb / 1024),
        samples_ms=durations,
        parameters_total=counts["total"],
        parameters_trainable=counts["trainable"],
        torch_threads=threads,
    )
