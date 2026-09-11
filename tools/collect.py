"""Record demonstrations with pixels for one silhouette, one episode file per seed.

Usage: uv run -m tools.collect --target house --episodes 20 --out data/house
       uv run -m tools.collect --target square --policy my_demo.py --fps 10

Each episode is a compressed npz holding top/wrist images, arm state, the
commanded action, piece poses, the goal and language annotations at `fps`
frames per second: the figure's task prompt, the demonstrator's numbered
`plan` (one sentence per step), and per frame the 1-based `step` in progress
and the fine-grained `subtask` sentence, read from `policy.plan`,
`policy.step` and `policy.subtask` when the demonstrator exposes them. Frames
are taken every 50/fps control ticks; the stored action is the last command of
that interval, so a learned policy that emits one action per frame and holds it
for the interval reproduces the demonstration. `tools/export.py` converts a
folder of episodes to a LeRobot dataset. Failed episodes are skipped unless
`--keep-failures`; every attempt is listed in `index.jsonl`.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark import CONTROL_SECONDS, DEFAULT_STEPS, HOLD_STEPS, PROTOCOL, SPLITS
from env import CAMERAS, IMAGE_SIZE, Env
from eval import load_policy
from shapes import TARGETS
from tangram import score

FORMAT = "tangram-episodes-v1"
CONTROL_HZ = round(1 / CONTROL_SECONDS)


class Recorder:
    """Accumulate frames for one episode; `save` writes one npz and returns its summary."""

    def __init__(self, env, seed, fps, prompt):
        if CONTROL_HZ % fps:
            raise ValueError(f"fps must divide {CONTROL_HZ}")
        self.env, self.seed, self.fps, self.prompt = env, seed, fps, prompt
        self.decimation = CONTROL_HZ // fps
        self.frames = {name: [] for name in ("state", "action", "tcp_pos", "pieces", "time")}
        self.subtasks = []  # Language annotation per frame; "" when the demonstrator has none.
        self.steps = []  # 1-based plan step per frame; 0 when the demonstrator has no plan.
        self.plan = []  # Numbered plan of the episode, one sentence per step.
        self.images = {name: [] for name in CAMERAS}
        self.pending = None  # Frame observation waiting for the interval's last action.
        self.latest = None  # Most recent rendered images, for policies that want pixels.

    def observe(self, obs, tick):
        """Call on every control tick before acting; renders only on frame ticks."""
        if tick % self.decimation == 0:
            self.flush(None)
            self.latest = self.env.images(0)
            self.pending = (obs, self.latest)

    def act(self, action, tick, policy=None):
        """Call with the command issued at `tick`; stores the frame at interval end."""
        if (tick + 1) % self.decimation == 0:
            self.flush(action, policy)

    def flush(self, action, policy=None):
        """Store the pending frame; a frame cut short by the episode end keeps its action."""
        if self.pending is None or action is None:
            self.pending = None
            return
        obs, images = self.pending
        self.subtasks.append(str(getattr(policy, "subtask", "")))
        self.steps.append(int(getattr(policy, "step", 0)))
        self.plan = [str(line) for line in getattr(policy, "plan", [])]
        self.frames["state"].append(np.asarray(obs["qpos"], dtype=np.float32))
        self.frames["action"].append(np.asarray(action, dtype=np.float32))
        self.frames["tcp_pos"].append(np.asarray(obs["tcp_pos"], dtype=np.float32))
        self.frames["pieces"].append(np.asarray(obs["pieces"], dtype=np.float32))
        self.frames["time"].append(np.float32(obs["time"]))
        for name in CAMERAS:
            self.images[name].append(images[name])
        self.pending = None

    def save(self, path, target, success, status, error, steps, goal):
        arrays = {key: np.stack(values) for key, values in self.frames.items()}
        arrays["subtask"] = np.asarray(self.subtasks, dtype=str)
        arrays["step"] = np.asarray(self.steps, dtype=np.int64)
        arrays["plan"] = np.asarray(self.plan, dtype=str)
        for name in CAMERAS:
            arrays[f"images_{name}"] = np.stack(self.images[name])
        with path.open("xb") as f:
            np.savez_compressed(
                f,
                **arrays,
                goal=np.asarray(goal, dtype=np.float32),
                seed=self.seed,
                target=target,
                prompt=self.prompt,
                robot=self.env.robot,
                fps=self.fps,
                success=bool(success),
                status=status,
                steps=steps,
                format=FORMAT,
                protocol=PROTOCOL,
                cameras=np.array(CAMERAS),
                image_size=np.array(IMAGE_SIZE),
            )
        return {
            "file": path.name,
            "seed": self.seed,
            "target": target,
            "success": bool(success),
            "status": status,
            "error": error,
            "frames": len(arrays["time"]),
            "steps": steps,
            "fps": self.fps,
        }


def record_episode(env, policy_cls, seed, target, prompt, fps, steps, stop_after_success=True):
    """Run one seeded episode and return (recorder, summary fields)."""
    obs = env.reset([seed], [target], prompt)[0]
    recorder = Recorder(env, seed, fps, obs["prompt"])
    policy = policy_cls(robot=env.robot, seed=seed)
    status, error, streak, tick = "completed", None, 0, 0
    action = np.r_[obs["qpos"][: env.narm], 1.0]
    for tick in range(steps):
        recorder.observe(obs, tick)
        try:
            # Pixels are at most one frame interval old, as they would be for a
            # chunked policy queried at the frame rate.
            action = policy.act({**obs, "images": recorder.latest})
            action = env.validate_actions(np.asarray(action, dtype=float).reshape(1, -1))[0]
        except Exception as exc:
            status, error = "policy_error", {"type": type(exc).__name__, "message": str(exc)}
            recorder.flush(action, policy)
            break
        recorder.act(action, tick, policy)
        obs = env.step([action])[0]
        streak = (
            streak + 1
            if score(obs["pieces"], obs["piece_velocities"], obs["goal"])["success"]
            else 0
        )
        # One extra second after the hold so the arm's retreat is part of the demonstration.
        if stop_after_success and streak >= HOLD_STEPS + CONTROL_HZ:
            recorder.flush(action, policy)
            tick += 1
            break
    else:
        recorder.flush(action, policy)
        tick = steps
    success = status == "completed" and streak >= HOLD_STEPS
    return recorder, success, status, error, tick, obs["goal"]


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--robot", choices=["panda", "piper"], default="panda")
    p.add_argument("--policy", type=Path, default=Path("examples/oracle.py"))
    p.add_argument("--split", choices=SPLITS, default="train")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--fps", type=int, default=10, help="Frame rate; must divide 50")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Horizon in control ticks")
    p.add_argument("--prompt", default=None, help="Override the per-figure task text")
    p.add_argument("--out", type=Path, help="Episode folder; default data/<target>")
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument(
        "--full-horizon", action="store_true", help="Do not stop early after a held success"
    )
    args = p.parse_args(argv)
    if min(args.episodes, args.fps, args.steps) < 1 or args.offset < 0:
        p.error("episodes, fps and steps must be positive; offset nonnegative")
    if CONTROL_HZ % args.fps:
        p.error(f"fps must divide the {CONTROL_HZ} Hz control rate")
    out = args.out or Path("data") / args.target
    out.mkdir(parents=True, exist_ok=True)
    policy_cls = load_policy(args.policy)
    env = Env(args.robot, pixels=True)
    seeds = range(
        SPLITS[args.split] + args.offset, SPLITS[args.split] + args.offset + args.episodes
    )
    kept = 0
    started = time.perf_counter()
    for seed in seeds:
        path = out / f"episode-{seed}.npz"
        if path.exists():
            print(json.dumps({"seed": seed, "skipped": "exists"}), flush=True)
            continue
        recorder, success, status, error, steps, goal = record_episode(
            env,
            policy_cls,
            seed,
            args.target,
            args.prompt,
            args.fps,
            args.steps,
            stop_after_success=not args.full_horizon,
        )
        row = {
            "seed": seed,
            "target": args.target,
            "success": success,
            "status": status,
            "error": error,
            "steps": steps,
            "frames": len(recorder.frames["time"]),
        }
        if success or args.keep_failures:
            row = recorder.save(path, args.target, success, status, error, steps, goal)
            kept += 1
        row["recorded_at"] = datetime.now(timezone.utc).isoformat()
        with (out / "index.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(row), flush=True)
    env.close()
    print(
        f"kept {kept}/{args.episodes} episodes in {out} ({time.perf_counter() - started:.0f} s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
