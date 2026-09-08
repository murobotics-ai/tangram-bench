"""Fixed evaluator: load a policy, run seeded episodes, write inspectable JSON.

Treat policy modules as trusted local code. This is not a security sandbox.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from env import DT, SUBSTEPS, Env
from tangram import score_batch
from tools.prepare import REVISION

PROTOCOL = "square-packed-state-v3"
SPLITS = {"train": 0, "dev": 100000, "test": 200000}


def load_policy(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("policy.py"))
    parser.add_argument("--robot", choices=["panda", "piper"], default="panda")
    parser.add_argument("--backend", choices=["cpu", "warp"], default="warp")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/result.json"))
    args = parser.parse_args()
    if (
        min(args.episodes, args.num_envs, args.steps) < 1
        or args.offset < 0
        or args.offset + args.episodes > 100000
    ):
        parser.error("positive episode/world/step counts required; seed range must fit its split")
    # Do not overwrite an experiment silently.
    if args.out.exists():
        parser.error(f"Output exists: {args.out}; choose another --out")
    root = Path(__file__).resolve().parent

    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(root), *cmd], text=True).strip()

    hashes = {
        p: digest(root / p)
        for p in ["eval.py", "env.py", "tangram.py", "tools/prepare.py", "uv.lock"]
    }
    policy_hash = digest(args.policy)
    git_commit, git_dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    policy_cls = load_policy(args.policy)
    started = time.perf_counter()
    rows = []
    inference = stepping = scoring = rollout = 0.0
    env = None
    for start in range(0, args.episodes, args.num_envs):
        seeds = list(
            range(
                SPLITS[args.split] + args.offset + start,
                SPLITS[args.split] + args.offset + min(start + args.num_envs, args.episodes),
            )
        )
        if env is None or env.num_envs != len(seeds):
            env = Env(args.robot, args.backend, len(seeds))
        observations = env.reset(seeds)
        policies = [policy_cls(robot=args.robot, seed=s) for s in seeds]
        streak = np.zeros(len(seeds), dtype=int)
        first_success = np.full(len(seeds), np.nan)
        loop_start = time.perf_counter()
        for step in range(args.steps):
            before = time.perf_counter()
            actions = [p.act(o) for p, o in zip(policies, observations)]
            inference += time.perf_counter() - before
            before = time.perf_counter()
            observations = env.step(actions)
            stepping += time.perf_counter() - before
            before = time.perf_counter()
            metrics = score_batch(
                np.stack([o["pieces"] for o in observations]),
                np.stack([o["piece_velocities"] for o in observations]),
                np.stack([o["goal"] for o in observations]),
            )
            scoring += time.perf_counter() - before
            streak = np.where(metrics["success"], streak + 1, 0)
            first_success[np.isnan(first_success) & (streak >= 25)] = (step + 1) * DT * SUBSTEPS
        rollout += time.perf_counter() - loop_start
        for j, seed in enumerate(seeds):
            metric = {name: values[j].item() for name, values in metrics.items()}
            # Final held success is the primary score. Transient success is diagnostic.
            metric["success"] = bool(streak[j] >= 25)
            rows.append(
                {
                    "seed": seed,
                    **metric,
                    "first_success_seconds": None
                    if np.isnan(first_success[j])
                    else float(first_success[j]),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    if hashes != {p: digest(root / p) for p in hashes} or policy_hash != digest(args.policy):
        raise RuntimeError("Source changed during evaluation; rerun with fixed files")
    packages = ["mujoco", "numpy", "shapely"] + (
        ["mujoco-warp", "warp-lang"] if args.backend == "warp" else []
    )
    model_bytes = np.empty(mujoco.mj_sizeModel(env.model), dtype=np.uint8)
    mujoco.mj_saveModel(env.model, buffer=model_bytes)
    result = {
        "protocol": PROTOCOL,
        "robot": args.robot,
        "backend": args.backend,
        "split": args.split,
        "steps": args.steps,
        "policy_hz": 1 / (DT * SUBSTEPS),
        "num_envs": args.num_envs,
        "policy": str(args.policy),
        "policy_sha256": policy_hash,
        "evaluator_sha256": hashes,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "platform": platform.platform(),
        "gpu": env.wp.get_device().name if args.backend == "warp" else None,
        "model_sha256": hashlib.sha256(model_bytes.tobytes()).hexdigest(),
        "menagerie_revision": REVISION,
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "wall_seconds_including_setup": time.perf_counter() - started,
        "policy_seconds_total": inference,
        "step_observe_seconds_total": stepping,
        "scoring_seconds_total": scoring,
        "rollout_seconds_total": rollout,
        "transitions_per_second": args.episodes * args.steps / rollout,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_iou": float(np.mean([r["iou"] for r in rows])),
        "episodes": rows,
        "note": "Prototype; seed holdout only, no unseen-shape or sim-to-real claim.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create protects results even if another run completed in the meantime.
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"success={result['success_rate']:.3f} mean_iou={result['mean_iou']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
