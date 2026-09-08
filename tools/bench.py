"""Measure the warmed evaluation loop, separately from setup and kernel compilation.

Uses the hold policy and exact scoring at every step. This measures engineering
throughput, not policy competence. Compare identical hardware, worlds and budgets.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from env import Env
from tangram import score_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["cpu", "warp"], default="warp")
    p.add_argument("--robot", choices=["panda", "piper"], default="panda")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("runs/perf.json"))
    args = p.parse_args()
    if min(args.num_envs, args.steps, args.repeat) < 1:
        p.error("worlds, steps and repeats must be positive")
    if args.out.exists():
        p.error(f"Output exists: {args.out}")
    setup_start = time.perf_counter()
    env = Env(args.robot, args.backend, args.num_envs)
    seeds = list(range(100000, 100000 + args.num_envs))
    obs = env.reset(seeds)  # Compiles and captures Warp kernels before timing.
    actions = np.stack([np.r_[o["qpos"][: env.narm], 1.0] for o in obs])
    for _ in range(20):
        obs = env.step(actions)
    setup_seconds = time.perf_counter() - setup_start
    samples = []
    for _ in range(args.repeat):
        obs = env.reset(seeds)
        actions = np.stack([np.r_[o["qpos"][: env.narm], 1.0] for o in obs])
        stepping = scoring = 0.0
        start = time.perf_counter()
        for _ in range(args.steps):
            before = time.perf_counter()
            obs = env.step(actions)
            stepping += time.perf_counter() - before  # Includes transfers and observation.
            before = time.perf_counter()
            score_batch(
                np.stack([o["pieces"] for o in obs]),
                np.stack([o["piece_velocities"] for o in obs]),
                np.stack([o["goal"] for o in obs]),
            )
            scoring += time.perf_counter() - before
        samples.append(
            {
                "seconds": time.perf_counter() - start,
                "step_seconds": stepping,
                "score_seconds": scoring,
            }
        )
    median = float(np.median([s["seconds"] for s in samples]))
    result = {
        "backend": args.backend,
        "robot": args.robot,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "setup_seconds": setup_seconds,
        "median_loop_seconds": median,
        "transitions_per_second": args.steps * args.num_envs / median,
        "gpu": env.wp.get_device().name if args.backend == "warp" else None,
        "platform": platform.platform(),
        "samples": samples,
        "scope": "Warmed fixed-action hold loop including observation transfer and per-step scoring; excludes reset.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        f"{result['transitions_per_second']:.0f} transitions/s | {median:.3f}s median | {args.out}"
    )


if __name__ == "__main__":
    main()
