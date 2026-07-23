#!/usr/bin/env python3
"""Benchmark inference speed with random dummy inputs — no real checkpoint needed
for the compile vs baseline comparison.

Runs three scenarios inside the same process:
  1. Baseline (eager) — COMPILE_MODEL=0
  2. torch.compile mode=default — COMPILE_MODEL=1
  3. torch.compile + CUDA graphs — COMPILE_MODEL=1, COMPILE_CUDAGRAPHS=1

Each reports mean/median/min/max/std latency across N bench cycles.

Usage:
    # Minimal smoke test (fast, ~30 s):
    CKPT=/path/to/checkpoint_unsharded python scripts/benchmark_inference.py

    # Full benchmark (30 warm + 30 bench per config, ~5 min):
    CKPT=/path/to/checkpoint_unsharded python scripts/benchmark_inference.py --warmup 30 --bench 30
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

import numpy as np
import torch


def make_dummy_inputs(agent, n_cams=4):
    """Generate random images + state that match the checkpoint's obs contract."""
    n_obs = agent.n_obs_steps
    H, W = 480, 640  # standard camera resolution
    images = [
        np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        for _ in range(n_obs * n_cams)
    ]
    state_dim = getattr(agent.model_config, "state_dim", 14)
    state = np.random.randn(n_obs, state_dim).astype(np.float32)
    return images, "Put the cube in the box.", state


def benchmark(agent, label, n_warmup, n_bench, save_actions=False):
    """Run a latency benchmark. Returns stats dict."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    images, task, state = make_dummy_inputs(agent)

    # Cold query (includes compile time if applicable)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    actions = agent.get_action_chunk(images=images, task_description=task, state=state)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000
    print(f"    cold query: {first_ms:.0f} ms   action shape: {actions.shape}")

    timings = []
    for i in range(n_warmup + n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        agent.get_action_chunk(images=images, task_description=task, state=state)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            timings.append(elapsed)

    arr = np.array(timings)
    stats = {
        "cold_ms": round(first_ms),
        "mean_ms": float(round(np.mean(arr), 1)),
        "median_ms": float(round(np.median(arr), 1)),
        "min_ms": float(round(np.min(arr), 1)),
        "max_ms": float(round(np.max(arr), 1)),
        "std_ms": float(round(np.std(arr), 1)),
        "n": n_bench,
    }
    print(f"    warm queries: {n_warmup}, bench queries: {n_bench}")
    print(f"    mean={stats['mean_ms']:.1f} ms, median={stats['median_ms']:.1f} ms, "
          f"min={stats['min_ms']:.1f} ms, max={stats['max_ms']:.1f} ms, "
          f"σ={stats['std_ms']:.1f} ms")
    return stats


def main():
    from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper

    parser = argparse.ArgumentParser(description="Benchmark MolmoBot inference speed")
    parser.add_argument("--gpu", default="0", help="CUDA device")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up queries per config")
    parser.add_argument("--bench", type=int, default=30, help="Bench queries per config")
    args = parser.parse_args()

    ckpt = os.environ.get("CKPT")
    if not ckpt:
        print("ERROR: Set CKPT=/path/to/checkpoint_unsharded")
        sys.exit(1)
    ckpt = os.path.abspath(ckpt)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    results = {}

    # ── 1. Baseline ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  STEP 1: Loading model (BASELINE — eager, no compile)")
    print(f"{'=' * 60}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    agent = SynthManipMolmoInferenceWrapper(ckpt, device="cuda", compile_model=False)
    load_s = round(time.perf_counter() - t0, 1)
    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"    load time: {load_s}s, peak GPU mem: {mem_gb:.1f} GB")
    results["baseline"] = benchmark(agent, "Baseline", args.warmup, args.bench)
    results["baseline"]["load_time_s"] = load_s
    results["baseline"]["peak_mem_gb"] = round(mem_gb, 2)
    del agent; gc.collect(); torch.cuda.empty_cache()

    # ── 2. torch.compile ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  STEP 2: Loading model (COMPILE — torch.compile mode=default)")
    print(f"{'=' * 60}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    agent = SynthManipMolmoInferenceWrapper(
        ckpt, device="cuda", compile_model=True, compile_mode="default"
    )
    load_s = round(time.perf_counter() - t0, 1)
    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"    load time: {load_s}s, peak GPU mem: {mem_gb:.1f} GB")
    results["compile"] = benchmark(agent, "torch.compile mode=default", args.warmup, args.bench)
    results["compile"]["load_time_s"] = load_s
    results["compile"]["peak_mem_gb"] = round(mem_gb, 2)
    del agent; gc.collect(); torch.cuda.empty_cache()

    # ── 3. torch.compile + CUDA graphs ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  STEP 3: Loading model (COMPILE + CUDA GRAPHS)")
    print(f"{'=' * 60}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    agent = SynthManipMolmoInferenceWrapper(
        ckpt, device="cuda", compile_model=True, compile_mode="default",
        compile_cudagraphs=True,
    )
    load_s = round(time.perf_counter() - t0, 1)
    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"    load time: {load_s}s, peak GPU mem: {mem_gb:.1f} GB")
    results["compile_cudagraphs"] = benchmark(
        agent, "torch.compile + CUDA graphs", args.warmup, args.bench
    )
    results["compile_cudagraphs"]["load_time_s"] = load_s
    results["compile_cudagraphs"]["peak_mem_gb"] = round(mem_gb, 2)
    del agent; gc.collect(); torch.cuda.empty_cache()

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    rows = []
    for cfg, r in results.items():
        speedup = results["baseline"]["mean_ms"] / r["mean_ms"] if r["mean_ms"] > 0 else 1.0
        rows.append((cfg, r, speedup))
        print(
            f"  {cfg:25s}  "
            f"mean={r['mean_ms']:7.1f} ms  "
            f"σ={r['std_ms']:5.1f} ms  "
            f"max={r['max_ms']:6.0f} ms  "
            f"mem={r.get('peak_mem_gb', '?'):4.1f} GB  "
            f"speedup={speedup:.2f}×"
        )
    print()

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "examples", "benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    entry = {
        "checkpoint": os.path.basename(ckpt),
        "torch_version": torch.__version__,
        **results,
    }
    try:
        old = json.load(open(out_path))
    except (FileNotFoundError, json.JSONDecodeError):
        old = []
    old.append(entry)
    json.dump(old, open(out_path, "w"), indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
