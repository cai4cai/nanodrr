"""Compare rendering backends on every (device, backend) pair this machine supports.

Pairs: torch on cpu/mps/cuda, plus the fused kernels (metal on MPS, triton on CUDA).
Reports accuracy against torch on the same device, speed, and working-set memory
(peak extra device memory while one render runs, on top of the resident volume).

Usage: python tests/benchmark/backends.py [--grad] [--labels] [--size 200 400] [--samples 500] [--batch 1]
"""

import argparse
import functools
import statistics
import time

import torch
from benchmark import PeakMemory, available_devices, device_name, setup_data, synchronize

from nanodrr.camera import make_k_inv
from nanodrr.drr import render


def available_backends(device: str) -> list[str]:
    """Backends runnable on `device`: torch everywhere, plus the fused kernel for that GPU."""
    backends = ["torch"]
    if device == "mps":
        try:
            from nanodrr.drr import _metal  # noqa: F401

            torch.mps.compile_shader  # noqa: B018
            backends.append("metal")
        except Exception:
            pass
    elif device == "cuda":
        try:
            import triton  # noqa: F401

            backends.append("triton")
        except ImportError:
            pass
    return backends


def pose_grad(args, backend):
    """Gradient of the summed render w.r.t. the pose (one forward + backward pass)."""
    subject, k_inv, rt_inv, *rest = args
    rt_inv = rt_inv.detach().clone().requires_grad_(True)
    render(subject, k_inv, rt_inv, *rest, backend=backend).sum().backward()
    return rt_inv.grad[:, :3]  # the homogeneous row's gradient is backend-specific


def working_set_mb(func, device: str) -> float:
    """Peak extra device memory (MB) allocated by one call of `func`."""
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        func()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 1024**2
    with PeakMemory(device) as tracker:
        func()
    synchronize(device)
    return tracker.peak / 1024**2


def time_us(func, device: str, runs: int, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            func()
        synchronize(device)
        times.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(times), statistics.pstdev(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--labels", action="store_true", help="Multi-class rendering with the demo labelmap")
    p.add_argument("--size", type=int, nargs="+", default=[100, 200, 400], help="Square detector sizes")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--grad", action="store_true", help="Time forward + backward w.r.t. the pose")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", nargs="+", default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--cpu-iterations", type=int, default=2, help="CPU is far slower, so loop less")
    a = p.parse_args()

    for device in a.device or available_devices():
        subject, _, rt_inv, sdd, _, _ = setup_data(a.image, "demo" if a.labels else None, device)
        rt_inv, sdd = rt_inv.expand(a.batch, -1, -1), sdd.expand(a.batch)
        print(
            f"\n### {device} ({device_name(device)}): volume {tuple(subject.image.shape[2:])}, "
            f"{subject.n_classes} class(es), B={a.batch}, S={a.samples}"
        )
        header = f"{'size':>5} {'backend':>7} {'ms/render':>10} {'±':>6} {'FPS':>7} {'speedup':>8} {'work MB':>8} {'rel err':>9}"
        print(header)

        for size in a.size:
            k = make_k_inv(1020.0, 400.0 / size, 400.0 / size, 0.0, 0.0, size, size)
            k = k.to(device).expand(a.batch, -1, -1)
            args = (subject, k, rt_inv, sdd, size, size, a.samples)
            ref = render(*args, backend="torch")
            if a.grad:
                ref = pose_grad(args, "torch")
            base_us = None
            for backend in available_backends(device):
                run = (
                    functools.partial(pose_grad, args, backend)
                    if a.grad
                    else functools.partial(render, *args, backend=backend)
                )

                err = ((run() - ref).abs().max() / ref.abs().max()).item()
                iters = a.cpu_iterations if device == "cpu" else a.iterations
                us, std = time_us(run, device, a.runs if device != "cpu" else 3, iters, 2 if device == "cpu" else 10)
                mem = working_set_mb(run, device)
                base_us = base_us or us
                print(
                    f"{size:>5} {backend:>7} {us / 1e3:>10.2f} {std / 1e3:>6.2f} {1e6 / us:>7.0f} "
                    f"{base_us / us:>7.1f}x {mem:>8.1f} {err:>9.1e}"
                )


if __name__ == "__main__":
    main()
