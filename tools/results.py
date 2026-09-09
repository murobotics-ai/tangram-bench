"""Verify saved trajectories or compare matched evaluations without running a policy.

Usage: uv run -m tools.results verify runs/result.json
       uv run -m tools.results compare runs/a.json runs/b.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import (
    CONTROL_SECONDS,
    PROTOCOL,
    SCHEMA_VERSION,
    STATE_FIELDS,
    compare_runs,
    digest,
    score_trajectory,
    summarize,
)
from tangram import goal


def verify(path):
    run = json.loads(path.read_text())
    if run.get("schema_version") != SCHEMA_VERSION or run.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported schema or protocol")
    if run.get("status") != "completed":
        raise ValueError("Run is incomplete or invalid; inspect its error and episode journal")
    if len(run["episodes"]) != run["requested_episodes"]:
        raise ValueError("Run has missing episodes")
    if sorted(row["seed"] for row in run["episodes"]) != sorted(run["seeds"]):
        raise ValueError("Episode seeds differ from the run manifest")
    if digest(path.parent / run["model"]) != run["model_sha256"]:
        raise ValueError("Compiled model hash mismatch")
    root = Path(__file__).resolve().parents[1]
    for source in ("benchmark.py", "tangram.py", "shapes.py"):
        if digest(root / source) != run["evaluator_sha256"][source]:
            raise ValueError(f"Scoring source changed: {source}; use the recorded revision")
    expected_targets = dict(zip(run["seeds"], run["targets"], strict=True))
    for row in run["episodes"]:
        if (
            "policy_audit" in row
            and digest(path.parent / row["policy_audit"]) != row["policy_audit_sha256"]
        ):
            raise ValueError("Policy audit hash mismatch")
        trace_path = path.parent / row["trajectory"]
        if digest(trace_path) != row["trajectory_sha256"]:
            raise ValueError(f"Trajectory hash mismatch: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as trace:
            if (
                trace["seed"].item() != row["seed"]
                or trace["protocol"].item() != PROTOCOL
                or trace["schema_version"].item() != SCHEMA_VERSION
                or trace["status"].item() != row["status"]
                or trace["target"].item() != row["target"]
                or row["target"] != expected_targets[row["seed"]]
                or trace["prompt"].item() != run["prompt"]
            ):
                raise ValueError("Trajectory identity mismatch")
            steps = row["steps_executed"]
            if not np.array_equal(trace["time"], np.arange(steps + 1) * CONTROL_SECONDS):
                raise ValueError("Trajectory time grid mismatch")
            if any(
                len(trace[key]) != steps + 1 or not np.isfinite(trace[key]).all()
                for key in STATE_FIELDS
            ):
                raise ValueError("Invalid observation evidence")
            if not np.array_equal(
                trace["goal"],
                np.broadcast_to(goal(row["seed"], row["target"]), trace["goal"].shape),
            ):
                raise ValueError("Goal does not match the seeded task")
            if len(trace["actions"]) != steps or len(trace["controls"]) != steps:
                raise ValueError("Trajectory length mismatch")
            if row["status"] == "completed" and steps != run["steps"]:
                raise ValueError("Completed episode did not run the full horizon")
            if row["status"] not in ("completed", "policy_error"):
                raise ValueError("Invalid episode status")
            scores = score_trajectory(trace, completed=row["status"] == "completed")
        for key, value in scores.items():
            if isinstance(value, float):
                equal = np.isclose(value, row[key], atol=1e-12, rtol=0)
            else:
                equal = value == row[key]
            if not equal:
                raise ValueError(f"Score mismatch: seed={row['seed']}, {key}")
    if summarize(run["episodes"]) != run["summary"]:
        raise ValueError("Summary mismatch")
    if run["success_rate"] != run["summary"]["success_rate"]:
        raise ValueError("Success rate mismatch")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify")
    check.add_argument("result", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            run = verify(args.result)
            output = {"verified": str(args.result), "summary": run["summary"]}
        else:
            output = compare_runs(verify(args.a), verify(args.b))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Invalid result: {exc}\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
