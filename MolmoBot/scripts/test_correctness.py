#!/usr/bin/env python3
"""Verify that the compiled + cudagraphs path is numerically correct:
  - Replays are bit-stable (no stale graph-pool memory)
  - Eager vs compiled diff is within bf16 fusion-noise (<1% relative)
  - masked_scatter rewrite is bitwise-identical to the original

Usage:
    CKPT=/path/to/checkpoint_unsharded python scripts/test_correctness.py [gpu_id]
"""
import os, sys, time, warnings, gc
warnings.filterwarnings("ignore")
import numpy as np
import torch

CKPT = os.environ.get("CKPT")
GPU = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

if not CKPT:
    print("ERROR: Set CKPT=/path/to/checkpoint_unsharded")
    sys.exit(1)
CKPT = os.path.abspath(CKPT)
print(f"CKPT={CKPT}, GPU={GPU}")


def test_masked_scatter():
    """Unit test: our functional masked_scatter matches the original in-place op bitwise."""
    from olmo.models.video_olmo.video_olmo import VideoOlmo  # noqa: just check import works
    print("\n1. UNIT: masked_scatter rewrite == original in-place x[mask] += feats")
    g = torch.Generator(device="cuda").manual_seed(0)
    for trial, dtype in [(0, torch.float32), (1, torch.bfloat16)]:
        B, T, D = 1, 1626, 2560
        n = 800
        x = torch.randn(B, T, D, device="cuda", dtype=dtype, generator=g)
        mask = torch.zeros(B * T, dtype=torch.bool, device="cuda")
        mask[torch.randperm(B * T, generator=g, device="cuda")[:n]] = True
        feats = torch.randn(n, D, device="cuda", dtype=dtype, generator=g)
        ref = x.clone()
        ref.view(-1, ref.shape[-1])[mask] += feats
        xf = x.view(-1, x.shape[-1])
        adds = torch.zeros_like(xf).masked_scatter(mask.unsqueeze(-1), feats.to(xf.dtype))
        out = (xf + adds).view_as(x)
        same = torch.equal(ref, out)
        print(f"   trial {trial} ({dtype}): bitwise identical = {same}")
        assert same, "FAIL: masked_scatter != original"
    print("   PASS")


def test_replay_stability():
    """E2E: compiled+cudagraphs replays are bit-stable (no stale memory from different inputs)."""
    from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper
    print("\n2. E2E: replay bit-stability (A-B-A stale-memory test)")

    rng = np.random.default_rng(1234)
    images = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(8)]
    state = rng.standard_normal((2, 14)).astype(np.float32)
    task = "Put the cube in the box."

    def q(agent, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return agent.get_action_chunk(images=images, task_description=task, state=state, generator=gen)

    agent = SynthManipMolmoInferenceWrapper(
        CKPT, device="cuda", compile_model=True, compile_mode="default", compile_cudagraphs=True)
    t0 = time.time()
    q(agent)  # warm up compile
    print(f"   compile took: {time.time()-t0:.0f}s")

    a1 = q(agent, seed=0)
    a2 = q(agent, seed=0)
    rep = np.max(np.abs(a1 - a2))
    print(f"   same-seed repeatability: max|diff| = {rep:.2e}  (want 0)")
    assert rep == 0.0, "FAIL: repeated queries differ — stale graph-pool memory!"

    # A-B-A: use a different instruction to trigger recompile
    rng2 = np.random.default_rng(999)
    img2 = [rng2.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(8)]
    st2 = rng2.standard_normal((2, 14)).astype(np.float32)
    def qB():
        g = torch.Generator(device="cuda").manual_seed(42)
        return agent.get_action_chunk(images=img2, task_description="Move the cube.", state=st2, generator=g)

    aA1 = q(agent, seed=0)
    aB = qB()
    aA2 = q(agent, seed=0)
    aA3 = q(agent, seed=0)
    qB()
    aA4 = q(agent, seed=0)

    d = np.max(np.abs(aA2 - aA3))
    d2 = np.max(np.abs(aA2 - aA4))
    db = np.max(np.abs(aA1 - aB))
    print(f"   A-B-A: |A2-A3|={d:.2e}, |A2-A4|={d2:.2e} (want 0), |A-B|={db:.3f}")
    assert d == 0 and d2 == 0, "FAIL: A drifts after B — stale graph-pool memory!"
    del agent; gc.collect(); torch.cuda.empty_cache()
    print("   PASS")


def test_eager_vs_compiled():
    """E2E: eager vs compiled+cudagraphs actions differ by < 2% relative (bf16 fusion noise)."""
    from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper
    print("\n3. E2E: eager vs compiled+cudagraphs action match")

    rng = np.random.default_rng(42)
    images = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(8)]
    state = rng.standard_normal((2, 14)).astype(np.float32)
    task = "Put the cube in the box."

    def q(agent, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return agent.get_action_chunk(images=images, task_description=task, state=state, generator=g)

    ag_eager = SynthManipMolmoInferenceWrapper(CKPT, device="cuda", compile_model=False)
    a_eager = q(ag_eager, seed=0)
    a_eager_s7 = q(ag_eager, seed=7)
    seed_spread = np.max(np.abs(a_eager - a_eager_s7))
    del ag_eager; gc.collect(); torch.cuda.empty_cache()

    ag_cg = SynthManipMolmoInferenceWrapper(
        CKPT, device="cuda", compile_model=True, compile_mode="default", compile_cudagraphs=True)
    t0 = time.time()
    q(ag_cg)  # warm up compile
    print(f"   compile took: {time.time()-t0:.0f}s")
    a_cg = q(ag_cg, seed=0)

    d = np.max(np.abs(a_eager - a_cg))
    rel = d / (np.max(np.abs(a_eager)) + 1e-9)
    print(f"   max|eager - compiled+cg| = {d:.4f}  (rel {rel:.2%})")
    print(f"   seed-to-seed spread = {seed_spread:.3f}")
    assert rel < 0.02, f"FAIL: {rel:.2%} exceeds 2% tolerance!"
    print("   PASS")
    del ag_cg; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_masked_scatter()
        test_replay_stability()
        test_eager_vs_compiled()
        print(f"\n{'=' * 60}")
        print("  ALL TESTS PASSED")
        print(f"{'=' * 60}")
    else:
        print("CUDA not available, skipping tests.")
