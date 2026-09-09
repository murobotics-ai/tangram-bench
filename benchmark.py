"""Evaluation contracts, trajectory scoring and matched-run statistics.

No policy or simulator is needed to score a stored trajectory.
"""

import hashlib
from collections import Counter, deque
from copy import deepcopy
from pathlib import Path

import numpy as np

from tangram import score_batch

PROTOCOL = "tangram-packed-state-v5"
SCHEMA_VERSION = 1
SPLITS = {"train": 0, "dev": 100000, "test": 200000}
CONTROL_SECONDS = 0.02
HOLD_STEPS = 25
DEFAULT_STEPS = 3000
STATE_FIELDS = ("qpos", "qvel", "tcp_pos", "tcp_mat", "pieces", "piece_velocities", "goal")


def digest(path):
    """Hash run artifacts without loading checkpoints or traces into memory."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class PolicyDriver:
    """Copy policy inputs/outputs and validate a whole chunk before its first command.

    Chunk lengths may vary from 1 to max_chunk. Commands are consumed in order;
    the policy sees a fresh observation only when the previous chunk is exhausted.
    """

    def __init__(self, policy, validate, max_chunk=1):
        self.policy, self.validate, self.max_chunk = policy, validate, max_chunk
        self.queue = deque()

    def act(self, observation):
        if not self.queue:
            action = np.asarray(self.policy.act(deepcopy(observation)), dtype=float)
            if action.ndim == 1:
                action = action[None]
            if action.ndim != 2 or not 1 <= len(action) <= self.max_chunk:
                raise ValueError(
                    f"Expected a single action or a chunk of 1..{self.max_chunk} actions"
                )
            self.queue.extend(self.validate(action))
        return self.queue.popleft().copy()


def score_trajectory(trace, *, completed=True):
    """Trace contains reset at index 0, then every post-action state (no subsampling)."""
    poses = np.asarray(trace["pieces"])
    if len(poses) == 0:
        raise ValueError("Trajectory must contain a reset state")
    metrics = score_batch(poses, trace["piece_velocities"], trace["goal"])
    streak, first = 0, None
    for step, success in enumerate(metrics["success"][1:], start=1):
        streak = streak + 1 if success else 0
        if streak >= HOLD_STEPS and first is None:
            first = step * CONTROL_SECONDS
    result = {name: values[-1].item() for name, values in metrics.items()}
    result["instantaneous_success"] = result["success"]
    result["success"] = bool(completed and streak >= HOLD_STEPS)
    result["first_success_seconds"] = first
    result["final_hold_steps"] = streak
    result["initial_iou"] = float(metrics["iou"][0])
    result["best_iou"] = float(np.max(metrics["iou"][1:])) if len(poses) > 1 else None
    result["steps_executed"] = len(poses) - 1
    result["failure_reasons"] = []
    if not result["success"]:
        if not completed:
            result["failure_reasons"].append("policy_error")
        for name in ("on_table", "flat", "still"):
            if not result[name]:
                result["failure_reasons"].append(name)
        if result["iou"] < 0.95:
            result["failure_reasons"].append("coverage")
        if result["overlap_fraction"] >= 0.01:
            result["failure_reasons"].append("overlap")
        if result["instantaneous_success"] and streak < HOLD_STEPS:
            result["failure_reasons"].append("insufficient_hold")
    return result


def wilson_interval(successes, total):
    """Two-sided 95% Wilson interval; finite even for zero or all successes."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Require 0 <= successes <= total and total >= 1")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [
        0.0 if successes == 0 else max(0.0, float(center - radius)),
        1.0 if successes == total else min(1.0, float(center + radius)),
    ]


def summarize(rows):
    if not rows or len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("Summary needs nonempty, unique scene seeds")
    if any(r["status"] not in ("completed", "policy_error") for r in rows):
        raise ValueError("Cannot summarize interrupted or invalid episodes")
    if any(r["status"] == "policy_error" and r["success"] for r in rows):
        raise ValueError("Policy errors cannot earn success")
    successes = sum(r["success"] for r in rows)
    completed = [r for r in rows if r["status"] == "completed"]
    return {
        "episode_count": len(rows),
        "completed_count": len(completed),
        "policy_error_count": len(rows) - len(completed),
        "success_rate": successes / len(rows),
        "success_rate_ci95": wilson_interval(successes, len(rows)),
        "mean_final_iou_completed": float(np.mean([r["iou"] for r in completed]))
        if completed
        else None,
        "failure_counts": dict(Counter(reason for r in rows for reason in r["failure_reasons"])),
    }


# Batch size is included conservatively: backend numerics and shared policy globals
# may depend on grouping. Policy/checkpoint hashes intentionally differ between runs.
COMPARISON_FIELDS = (
    "schema_version",
    "protocol",
    "robot",
    "backend",
    "split",
    "steps",
    "policy_hz",
    "num_envs",
    "max_chunk",
    "prompt",
    "targets",
    "dataset_version",
    "max_inference_calls",
    "max_output_tokens",
    "policy_access",
    "evaluator_sha256",
    "model_sha256",
    "packages",
    "python",
)


def compare_runs(a, b):
    """Paired comparison on exactly the same seeds; no silently dropped failures."""
    for run in (a, b):
        if run.get("status") != "completed":
            raise ValueError("Only completed runs can be compared")
        if run.get("schema_version") != SCHEMA_VERSION or run.get("protocol") != PROTOCOL:
            raise ValueError("Unsupported schema or protocol")
    for key in COMPARISON_FIELDS:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"Incompatible runs: {key}")
    rows = []
    for run in (a, b):
        episodes = run["episodes"]
        if len(episodes) != run["requested_episodes"]:
            raise ValueError("Run has missing episodes")
        summarize(episodes)  # Checks duplicate seeds too.
        rows.append({r["seed"]: r for r in episodes})
    if rows[0].keys() != rows[1].keys():
        raise ValueError("Runs must use exactly the same seeds")
    seeds = sorted(rows[0])
    delta = np.array([int(rows[1][s]["success"]) - int(rows[0][s]["success"]) for s in seeds])
    # Resample paired scenes, not the two runs independently. Fixed RNG for reproducibility.
    rng = np.random.default_rng(0)
    means = np.array([np.mean(rng.choice(delta, len(delta))) for _ in range(10000)])
    return {
        "episodes": len(seeds),
        "a": summarize(a["episodes"]),
        "b": summarize(b["episodes"]),
        "success_rate_difference_b_minus_a": float(delta.mean()),
        "paired_bootstrap_ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "b_wins": int((delta > 0).sum()),
        "a_wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "note": "Paired scene bootstrap, 10000 resamples, seed 0. Intervals can degenerate on "
        "constant outcomes; they do not establish equivalence or account for training variance.",
    }
