"""Run fixed seeded episodes with failure accounting and replayable scoring evidence.

Policy modules are trusted local code, not a security sandbox. Results never overwrite.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from adapters import policy_factory, system_config
from benchmark import (
    CONTROL_SECONDS,
    DEFAULT_STEPS,
    PROTOCOL,
    SCHEMA_VERSION,
    SPLITS,
    STATE_FIELDS,
    PolicyDriver,
    digest,
    score_trajectory,
    summarize,
)
from env import CAMERAS, DT, IMAGE_SIZE, SUBSTEPS, Env
from shapes import SUITES, TARGETS, VERSION, prompt
from tools.prepare import REVISION


def load_policy(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def error_record(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def run_batch(
    env,
    policy_cls,
    seeds,
    steps,
    max_chunk,
    save_episode,
    targets=None,
    prompt=None,
    max_inference_calls=None,
):
    """Keep failed policies out of later inference; preserve every attempted episode.

    An environment failure aborts the batch/run. A policy failure earns zero and
    its world holds the last valid command while unaffected worlds finish.
    """
    observations = env.reset(seeds, targets, prompt)
    if env.pixels:
        # Pixels are rendered on demand, only for the world and tick that asks the policy.
        for j, obs in enumerate(observations):
            obs["images"] = env.images(j)
    traces = [{key: [obs[key].copy()] for key in STATE_FIELDS} for obs in observations]
    subtasks = [[] for _ in seeds]  # Optional language annotation a policy exposes per tick.
    steps_done = [[] for _ in seeds]
    actions = [[] for _ in seeds]
    controls = [[] for _ in seeds]
    drivers = [None for _ in seeds]
    errors = [None for _ in seeds]
    status = ["completed" for _ in seeds]
    latency = [[] for _ in seeds]
    previous = [np.r_[o["qpos"][: env.narm], 1.0] for o in observations]
    fatal = None
    try:
        for j, seed in enumerate(seeds):
            try:
                drivers[j] = PolicyDriver(
                    policy_cls(robot=env.robot, seed=seed), env.validate_actions, max_chunk
                )
            except Exception as exc:
                errors[j], status[j] = error_record(exc), "policy_error"
        for _ in range(steps):
            active = [j for j in range(len(seeds)) if status[j] == "completed"]
            for j in active:
                before = time.perf_counter()
                inference = not drivers[j].queue
                try:
                    # Only inference calls contribute to latency, not queued commands.
                    if (
                        inference
                        and max_inference_calls is not None
                        and len(latency[j]) >= max_inference_calls
                    ):
                        inference = False
                        raise RuntimeError("Inference call budget exhausted")
                    if inference and env.pixels and "images" not in observations[j]:
                        observations[j]["images"] = env.images(j)
                    previous[j] = drivers[j].act(observations[j])
                except Exception as exc:
                    errors[j], status[j] = error_record(exc), "policy_error"
                finally:
                    if inference:
                        latency[j].append(time.perf_counter() - before)
            if all(s != "completed" for s in status):
                break
            observations = env.step(previous)
            for j, obs in enumerate(observations):
                if status[j] == "completed":
                    for key in STATE_FIELDS:
                        traces[j][key].append(obs[key].copy())
                    actions[j].append(previous[j].copy())
                    controls[j].append(env.previous[j].copy())
                    subtasks[j].append(str(getattr(drivers[j].policy, "subtask", "")))
                    steps_done[j].append(int(getattr(drivers[j].policy, "step", 0)))
    except (Exception, KeyboardInterrupt) as exc:
        fatal = exc
        for j in range(len(seeds)):
            if status[j] == "completed":
                errors[j] = error_record(exc)
                status[j] = (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "environment_error"
                )
    finally:
        for j, seed in enumerate(seeds):
            trace = {key: np.stack(values) for key, values in traces[j].items()}
            trace["actions"] = np.asarray(actions[j]).reshape(-1, env.narm + 1)
            trace["controls"] = np.asarray(controls[j]).reshape(-1, env.narm + 1)
            trace["time"] = np.arange(len(trace["pieces"])) * CONTROL_SECONDS
            trace["subtask"] = np.asarray(subtasks[j], dtype=str)
            trace["step"] = np.asarray(steps_done[j], dtype=np.int64)
            plan = getattr(drivers[j].policy, "plan", []) if drivers[j] else []
            trace["plan"] = np.asarray([str(line) for line in plan], dtype=str)
            trace["prompt"] = env.prompts[j]
            row = {"seed": seed, "target": env.targets[j], "status": status[j], "error": errors[j]}
            row["policy_access"] = (
                getattr(drivers[j].policy, "access", "state") if drivers[j] else "state"
            )
            if status[j] in ("completed", "policy_error"):
                row.update(score_trajectory(trace, completed=status[j] == "completed"))
            else:
                row.update(success=False, steps_executed=len(trace["actions"]))
            row["inference_calls"] = len(latency[j])
            row["inference_seconds_total"] = float(sum(latency[j]))
            row["inference_seconds_p50"] = float(np.median(latency[j])) if latency[j] else None
            row["inference_seconds_p95"] = (
                float(np.quantile(latency[j], 0.95)) if latency[j] else None
            )
            if drivers[j] is not None and hasattr(drivers[j].policy, "audit"):
                try:
                    row["policy_audit"] = drivers[j].policy.audit()
                except Exception as exc:
                    row["audit_error"] = error_record(exc)
            save_episode(row, trace)
    if fatal is not None:
        raise fatal


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--policy", type=Path, default=Path("policy.py"))
    source.add_argument("--system", type=Path, help="Adapter/checkpoint JSON configuration")
    p.add_argument("--robot", choices=["panda", "piper"], default="panda")
    p.add_argument("--backend", choices=["cpu", "warp"], default="cpu")
    p.add_argument(
        "--obs",
        choices=["state", "pixels"],
        default="state",
        help="pixels adds observation['images'] (top and wrist, uint8) on every inference",
    )
    p.add_argument("--episodes", type=int, default=4, help="Unique scenes; 4 is a smoke test")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--split", choices=SPLITS, default="dev")
    p.add_argument("--target", choices=(*TARGETS, "suite"), default="suite")
    p.add_argument(
        "--prompt", default=None, help="Override the per-figure task text for every episode"
    )
    p.add_argument("--max-inference-calls", type=int, default=DEFAULT_STEPS)
    p.add_argument(
        "--max-output-tokens", type=int, default=4096, help="Per-request ceiling for model adapters"
    )
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-chunk", type=int, default=1, help="Maximum open-loop action chunk length")
    p.add_argument(
        "--policy-metadata", type=Path, help="JSON object: data, training, compute, model config"
    )
    p.add_argument(
        "--policy-artifact",
        type=Path,
        action="append",
        default=[],
        help="Hash a checkpoint or imported source; repeat for multiple files",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/runs/result.json"))
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (
        min(
            args.episodes,
            args.num_envs,
            args.steps,
            args.max_chunk,
            args.max_inference_calls,
            args.max_output_tokens,
        )
        < 1
        or args.offset < 0
        or args.offset + args.episodes > 100000
    ):
        p.error("Positive episode/world/step/chunk counts required; seeds must fit their split")
    if args.out.exists():
        p.error(f"Output exists: {args.out}; choose another --out")
    metadata = json.loads(args.policy_metadata.read_text()) if args.policy_metadata else {}
    if not isinstance(metadata, dict):
        p.error("Policy metadata must be a JSON object")
    json.dumps(metadata, allow_nan=False)
    assert DT * SUBSTEPS == CONTROL_SECONDS
    root = Path(__file__).resolve().parent
    sources = [
        "eval.py",
        "benchmark.py",
        "env.py",
        "tangram.py",
        "shapes.py",
        "adapters.py",
        "tools/prepare.py",
        "uv.lock",
    ]
    hashes = {path: digest(root / path) for path in sources}
    artifact_files = [args.policy, *args.policy_artifact]
    system = None
    if args.system:
        system, system_files = system_config(args.system)
        artifact_files = [*system_files, *args.policy_artifact]
        metadata = {**metadata, "system": system}
        if system.get("chunk_size", 1) > args.max_chunk:
            p.error("System chunk_size exceeds --max-chunk")
        if system.get("max_output_tokens", 4096) > args.max_output_tokens:
            p.error("System max_output_tokens exceeds the evaluation limit")
    if args.policy_metadata:
        artifact_files.append(args.policy_metadata)
    artifacts = {str(path.resolve()): digest(path) for path in artifact_files}

    def git(*cmd):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *cmd], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = ["mujoco", "numpy", "shapely"] + (
        ["mujoco-warp", "warp-lang"] if args.backend == "warp" else []
    )
    git_status = git("status", "--porcelain")
    suite = SUITES[args.split] if args.target == "suite" else (args.target,)
    # Episodes cycle through the suite's figures; use a multiple of len(suite)
    # for equal counts per figure in reported runs.
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "robot": args.robot,
        "backend": args.backend,
        "observation": args.obs,
        "cameras": list(CAMERAS) if args.obs == "pixels" else [],
        "image_size": list(IMAGE_SIZE) if args.obs == "pixels" else None,
        "split": args.split,
        "steps": args.steps,
        "policy_hz": 1 / CONTROL_SECONDS,
        "num_envs": args.num_envs,
        "max_chunk": args.max_chunk,
        "max_inference_calls": args.max_inference_calls,
        "max_output_tokens": args.max_output_tokens,
        "prompt": args.prompt,
        "prompts": {t: args.prompt if args.prompt is not None else prompt(t) for t in suite},
        "dataset_version": VERSION,
        "targets": [suite[i % len(suite)] for i in range(args.episodes)],
        "requested_episodes": args.episodes,
        "seeds": list(
            range(
                SPLITS[args.split] + args.offset, SPLITS[args.split] + args.offset + args.episodes
            )
        ),
        "policy": str((args.system or args.policy).resolve()),
        "policy_sha256": artifacts[str((args.system or args.policy).resolve())],
        "policy_artifacts": artifacts,
        "policy_metadata": metadata,
        "evaluator_sha256": hashes,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "menagerie_revision": REVISION,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "episodes": [],
        "note": "Public pilot corpus; the held-out figure is public, not secret. "
        "Policy errors count as failures. No sim-to-real claim.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    directory = args.out.with_suffix(".artifacts")
    try:
        directory.mkdir()  # Exclusive reservation, including concurrent runs.
    except FileExistsError:
        p.error(f"Artifacts already exist: {directory}; choose another --out")
    write_json(directory / "manifest.json", result)
    started = time.perf_counter()

    def save_episode(row, trace):
        name = f"seed-{row['seed']}.npz"
        with (directory / name).open("xb") as f:
            np.savez_compressed(
                f,
                **trace,
                seed=row["seed"],
                protocol=PROTOCOL,
                schema_version=SCHEMA_VERSION,
                status=row["status"],
                target=row["target"],
            )
        row["trajectory"] = str(Path(directory.name) / name)
        row["trajectory_sha256"] = digest(directory / name)
        if "policy_audit" in row:
            audit = row.pop("policy_audit")
            audit_name = f"seed-{row['seed']}-policy.json"
            write_json(directory / audit_name, audit)
            row["policy_usage"] = audit.get("usage", {})
            row["policy_audit"] = str(Path(directory.name) / audit_name)
            row["policy_audit_sha256"] = digest(directory / audit_name)
        with (directory / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        result["episodes"].append(row)
        print(json.dumps(row), flush=True)

    exit_code = 0
    try:
        policy_cls = policy_factory(system) if system else load_policy(args.policy)
        env = None
        for start in range(0, args.episodes, args.num_envs):
            seeds = result["seeds"][start : start + args.num_envs]
            if env is None or env.num_envs != len(seeds):
                if env is not None:
                    env.close()
                env = Env(args.robot, args.backend, len(seeds), pixels=args.obs == "pixels")
                model = np.empty(mujoco.mj_sizeModel(env.model), dtype=np.uint8)
                mujoco.mj_saveModel(env.model, buffer=model)
                model_hash = hashlib.sha256(model.tobytes()).hexdigest()
                if "model_sha256" in result and result["model_sha256"] != model_hash:
                    raise RuntimeError("Compiled model changed between batches")
                result["model_sha256"] = model_hash
                result["model"] = str(Path(directory.name) / "model.mjb")
                if not (directory / "model.mjb").exists():
                    with (directory / "model.mjb").open("xb") as f:
                        f.write(model.tobytes())
                result["gpu"] = env.wp.get_device().name if args.backend == "warp" else None
            run_batch(
                env,
                policy_cls,
                seeds,
                args.steps,
                args.max_chunk,
                save_episode,
                result["targets"][start : start + len(seeds)],
                args.prompt,
                args.max_inference_calls,
            )
        if hashes != {path: digest(root / path) for path in hashes} or artifacts != {
            path: digest(path) for path in artifacts
        }:
            raise RuntimeError("Source/checkpoint changed during evaluation; run is invalid")
        env.close()
        result["status"] = "completed"
        result["policy_access"] = sorted({row["policy_access"] for row in result["episodes"]})
        result["summary"] = summarize(result["episodes"])
        result["success_rate"] = result["summary"]["success_rate"]
    except (Exception, KeyboardInterrupt) as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        result["status"] = "interrupted" if exit_code == 130 else "error"
        result["error"] = error_record(exc)
        result["summary"] = None  # Never silently average an incomplete/invalid run.
        result["success_rate"] = None
    result["wall_seconds_including_setup"] = time.perf_counter() - started
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.out, result)
    print(f"{result['status']}: success={result['success_rate']} -> {args.out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
