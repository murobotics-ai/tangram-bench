"""Record demonstrations with pixels for one silhouette, one episode file per seed.

Usage: uv run -m tools.collect --target house --episodes 20 --out data/house
       uv run -m tools.collect --target house --episodes 60 --workers 10 --watch
       uv run -m tools.collect --target square --policy my_demo.py --fps 10

Each episode is a compressed npz holding top/wrist images, arm state, the
commanded action, piece poses, the goal and language annotations at `fps`
frames per second: the figure's task prompt, the demonstrator's numbered
`plan` (one sentence per step), and per frame the 1-based `step` in progress
and the fine-grained `subtask` sentence, read from `policy.plan`,
`policy.step` and `policy.subtask` when the demonstrator exposes them. Frames
are taken every 50/fps control ticks; the stored action is the last command of
that interval. A learned policy that emits one action per frame and holds it for
the interval follows the demonstration approximately, not exactly: the
demonstrator may change its command inside the interval and the 2 rad/s slew
limit depends on the sequence (replaying held actions from the same reset
drifts by up to 0.05 rad over 300 ticks). Exact 50 Hz records come from eval.py. `tools/export.py` converts a
folder of episodes to a LeRobot dataset. Failed episodes are skipped unless
`--keep-failures`; every attempt is listed in `index.jsonl`.

Every launch is a round: its episodes go to `<out>/<YYYY-MM-DD-HH-MM-SS>/`
(local time; `--round NAME` chooses the folder). `--resume` continues the
newest round (or the one named by `--round`) with `--episodes` more seeds,
starting after the last seed the round lists, the way `lerobot-record
--resume` grows a dataset; `--offset` overrides the start and seeds whose file
already exists are skipped, so a round can be topped up or retried
indefinitely. `tools.export` and
`tools.report` take the figure folder and read every round in it, the newest
copy of a seed winning; `view.py --demo <round>/episode-SEED.npz` replays one.

`--workers N` records N seeds at a time, one process, arm and table each, with
headless EGL rendering; the episodes are identical to a sequential run because
a seed fixes the scene, the goal and the demonstrator. `--watch` opens one 3D
scene with every arm on its own table, like a multi-robot RL arena, with each
arm's seed, progress and current subtask listed; `--watch grid` tiles one
camera per arm instead (`--watch-camera context|top|wrist`). Physics stays in
the workers, the window only draws, and closing it leaves the collection running;
once every arm is done it stays open on the final state until closed, so chain
unattended runs without `--watch`.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from benchmark import CONTROL_SECONDS, DEFAULT_STEPS, HOLD_STEPS, PROTOCOL, SPLITS
from env import CAMERAS, IMAGE_SIZE, Env, draw_goal, fleet_model
from eval import load_policy
from shapes import TARGETS
from tangram import SPLIT_SIZE, describe_layout, score

ROUND = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")
FINISHED = "All arms done, episodes saved. Esc or close the window to finish"


def progress(row, done, total):
    """One human line per finished episode, next to the JSON row."""
    first = row.get("first_success_step", -1)
    seconds = (first if first >= 0 else row["steps"]) / CONTROL_HZ
    if row["success"]:
        outcome = f"success at {seconds:.0f} s"
    elif row["status"] == "completed":
        outcome = f"no success in {seconds:.0f} s"
    else:
        outcome = f"{row['status']} at {seconds:.0f} s"
    return f"[{done}/{total}] seed {row['seed']} {outcome}"


FORMAT = "tangram-episodes"
CONTROL_HZ = round(1 / CONTROL_SECONDS)


class Recorder:
    """Accumulate frames for one episode; `save` writes one npz and returns its summary."""

    def __init__(self, env, seed, fps, prompt, steps=DEFAULT_STEPS):
        if CONTROL_HZ % fps:
            raise ValueError(f"fps must divide {CONTROL_HZ}")
        self.env, self.seed, self.fps, self.prompt = env, seed, fps, prompt
        self.decimation = CONTROL_HZ // fps
        # Frames live in preallocated arrays: appending to lists and stacking at save
        # time doubles the memory of a full-horizon episode (about 1.1 GB of images),
        # which with ten workers finishing together pushes the machine into swap.
        capacity = steps // self.decimation + 2
        narm = env.narm
        self.frames = {
            "state": np.empty((capacity, narm + 2), dtype=np.float32),
            "action": np.empty((capacity, narm + 1), dtype=np.float32),
            "tcp_pos": np.empty((capacity, 3), dtype=np.float32),
            "pieces": np.empty((capacity, 7, 7), dtype=np.float32),
            "time": np.empty(capacity, dtype=np.float32),
        }
        self.images = {
            name: np.empty((capacity, *IMAGE_SIZE, 3), dtype=np.uint8) for name in CAMERAS
        }
        self.count = 0  # Frames stored so far.
        self.subtasks = []  # Language annotation per frame; "" when the demonstrator has none.
        self.steps = []  # 1-based plan step per frame; 0 when the demonstrator has no plan.
        self.plan = []  # Numbered plan of the episode, one sentence per step.
        self.pending = None  # Frame observation waiting for the interval's last action.
        self.latest = None  # Most recent rendered images, for policies that want pixels.
        self.scene = {}  # Designed layout of the episode (degrees, millimetres).

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
        if self.count >= len(self.frames["time"]):
            self.grow()  # A teleoperation session has no fixed horizon.
        i = self.count
        self.subtasks.append(str(getattr(policy, "subtask", "")))
        self.steps.append(int(getattr(policy, "step", 0)))
        self.plan = [str(line) for line in getattr(policy, "plan", [])]
        self.frames["state"][i] = obs["qpos"]
        self.frames["action"][i] = action
        self.frames["tcp_pos"][i] = obs["tcp_pos"]
        self.frames["pieces"][i] = obs["pieces"]
        self.frames["time"][i] = obs["time"]
        for name in CAMERAS:
            self.images[name][i] = images[name]
        self.count += 1
        self.pending = None

    def grow(self):
        """Double the frame capacity; the preallocation only sizes the common case."""
        for table in (self.frames, self.images):
            for key, array in table.items():
                table[key] = np.concatenate([array, np.empty_like(array)])

    def save(self, path, target, success, status, error, steps, goal, first_success=-1):
        n = self.count
        arrays = {key: values[:n] for key, values in self.frames.items()}
        arrays["subtask"] = np.asarray(self.subtasks, dtype=str)
        arrays["step"] = np.asarray(self.steps, dtype=np.int64)
        arrays["plan"] = np.asarray(self.plan, dtype=str)
        for name in CAMERAS:
            arrays[f"images_{name}"] = self.images[name][:n]
        # Write to a temporary name and rename: a crash never leaves a partial episode
        # that a later run would mistake for a finished one.
        partial = path.with_name(path.name + ".partial")
        with partial.open("wb") as f:
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
                first_success_step=int(first_success),
                scene=json.dumps(self.scene),
                format=FORMAT,
                protocol=PROTOCOL,
                cameras=np.array(CAMERAS),
                image_size=np.array(IMAGE_SIZE),
            )
        os.replace(partial, path)
        return {
            "file": path.name,
            "seed": self.seed,
            "target": target,
            "success": bool(success),
            "status": status,
            "error": error,
            "frames": n,
            "steps": steps,
            "first_success_step": int(first_success),
            "fps": self.fps,
            **self.scene,
        }


def record_episode(
    env,
    policy_cls,
    seed,
    target,
    prompt,
    fps,
    steps,
    stop_after_success=True,
    on_frame=None,
    scene=None,
):
    """Run one seeded episode and return (recorder, summary fields).

    `on_frame(tick, images, policy)` is called on every frame tick with the fresh
    render, for live monitoring; it never touches what gets recorded.
    """
    obs = env.reset([seed], [target], prompt, scenes=[scene] if scene else None)[0]
    recorder = Recorder(env, seed, fps, obs["prompt"], steps)
    recorder.scene = describe_layout(env.scenes[0])
    policy = policy_cls(robot=env.robot, seed=seed)
    status, error, streak, tick = "completed", None, 0, 0
    first_success = -1  # Control step at which the success test first held for HOLD_STEPS.
    action = np.r_[obs["qpos"][: env.narm], 1.0]
    for tick in range(steps):
        recorder.observe(obs, tick)
        if on_frame is not None and tick % recorder.decimation == 0:
            on_frame(tick, recorder.latest, policy)
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
        if streak == HOLD_STEPS and first_success < 0:
            first_success = tick + 1
        # Four more seconds after the hold: the retreat and the return home are demonstrated.
        if stop_after_success and streak >= HOLD_STEPS + 4 * CONTROL_HZ:
            recorder.flush(action, policy)
            tick += 1
            break
    else:
        recorder.flush(action, policy)
        tick = steps
    success = status == "completed" and streak >= HOLD_STEPS
    return recorder, success, status, error, tick, obs["goal"], first_success


def episode_row(recorder, success, status, error, steps, target, seed, first_success=-1):
    return {
        "seed": seed,
        "target": target,
        "success": success,
        "status": status,
        "error": error,
        "steps": steps,
        "frames": recorder.count,
        "first_success_step": int(first_success),
    }


def round_name(now=None):
    """Folder name of one launch: the local date and time, hyphen separated."""
    return (now or datetime.now()).strftime("%Y-%m-%d-%H-%M-%S")


def latest_round(figure_folder):
    """The newest dated round folder under a figure folder, or None."""
    rounds = sorted(
        p for p in Path(figure_folder).glob("*") if p.is_dir() and ROUND.fullmatch(p.name)
    )
    return rounds[-1] if rounds else None


def recorded_seeds(round_folder):
    """Seeds listed in a round's index, attempted or saved."""
    index = Path(round_folder) / "index.jsonl"
    if not index.exists():
        return set()
    return {json.loads(line)["seed"] for line in index.read_text().splitlines() if line.strip()}


def append_index(out, row):
    row["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with (out / "index.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row), flush=True)


def record_one(env, policy_cls, seed, args, out, on_frame=None):
    """Record one seed and return its index row and whether a file was kept.

    The recorder (about 1.5 GB at the full horizon) is local to this call, so it
    is freed before the next episode allocates its own; the collector's memory
    is one episode per worker, not two.
    """
    path = out / f"episode-{seed}.npz"
    scene = {
        "goal_yaw": np.radians(args.goal_yaw) if args.goal_yaw is not None else None,
        "source_yaw": np.radians(args.source_yaw) if args.source_yaw is not None else None,
    }
    recorder, success, status, error, steps, goal, first = record_episode(
        env,
        policy_cls,
        seed,
        args.target,
        args.prompt,
        args.fps,
        args.steps,
        stop_after_success=not args.full_horizon,
        on_frame=(lambda tick, images, policy: on_frame(seed, tick, images, policy))
        if on_frame
        else None,
        scene=scene if any(v is not None for v in scene.values()) else None,
    )
    row = episode_row(recorder, success, status, error, steps, args.target, seed, first)
    row.update(recorder.scene)
    if success or args.keep_failures:
        if on_frame is not None:
            on_frame(seed, -1, None, None)  # Tell the watch window the file is being written.
        row = recorder.save(path, args.target, success, status, error, steps, goal, first)
        return row, True
    return row, False


def record_seeds(env, policy_cls, seeds, args, out, on_frame=None, on_row=None):
    """Record `seeds` in order on one environment; returns the number of episodes kept."""
    kept = 0
    for seed in seeds:
        row, saved = record_one(env, policy_cls, seed, args, out, on_frame)
        kept += saved
        (on_row or (lambda r: append_index(out, r)))(row)
    return kept


def worker(index, seeds, args, out, queue, frame_name, state_name):
    """One process, one arm and table: records its seeds and publishes to the watch window."""
    from multiprocessing import shared_memory

    frame = shared_memory.SharedMemory(name=frame_name)
    shared = np.ndarray((*IMAGE_SIZE, 3), dtype=np.uint8, buffer=frame.buf)
    state = shared_memory.SharedMemory(name=state_name)
    env = None

    def on_frame(seed, tick, images, policy):
        if tick < 0:
            queue.put({"worker": index, "seed": seed, "saving": True})
            return
        if args.watch == "grid":
            # The grid may show a camera the dataset does not record, e.g. the context view.
            camera = args.watch_camera
            shared[:] = images[camera] if camera in images else env.render(camera)
        else:
            np.ndarray(env.model.nq, dtype=np.float64, buffer=state.buf)[:] = env.data[0].qpos
        message = {
            "worker": index,
            "seed": seed,
            "tick": tick,
            "label": str(getattr(policy, "subtask", "")),
            "step": int(getattr(policy, "step", 0)),
            "plan": len(getattr(policy, "plan", [])),
        }
        if tick == 0:
            message["goal"] = np.asarray(env.outlines[0]).tolist()
        queue.put(message)

    try:
        env = Env(args.robot, pixels=True)
        policy_cls = load_policy(args.policy)
        kept = record_seeds(
            env,
            policy_cls,
            seeds,
            args,
            out,
            on_frame if args.watch else None,
            lambda row: queue.put({"worker": index, "row": row}),
        )
        env.close()
        queue.put({"worker": index, "done": kept})
    except BaseException as exc:  # Report, so the parent never waits on a dead worker.
        queue.put({"worker": index, "error": f"{type(exc).__name__}: {exc}", "done": 0})
        raise
    finally:
        frame.close()
        state.close()


class Fleet:
    """One 3D scene with every worker's arm and table, like a multi-robot RL arena.

    Physics stays in the workers; they publish `qpos` and this window only draws.
    Drag to orbit, right-drag to pan, scroll to zoom, Esc to close.
    """

    def __init__(self, states, robot, target, steps):
        import glfw

        self.glfw, self.states, self.steps = glfw, states, steps
        self.finished = False
        self.status = [
            {"seed": None, "tick": 0, "label": "", "step": 0, "plan": 0, "done": None}
            for _ in states
        ]
        self.goals = [None] * len(states)
        columns = next((c for c in range(min(len(states), 5), 1, -1) if len(states) % c == 0), 5)
        self.model, self.offsets, self.index, self.shift = fleet_model(robot, len(states), columns)
        self.data = mujoco.MjData(self.model)
        if not glfw.init():
            raise RuntimeError(
                "Cannot open a display for --watch; drop the flag to record headless"
            )
        self.window = glfw.create_window(
            1440, 900, f"Tangram-Bench | {len(states)} arms collecting {target}", None, None
        )
        if not self.window:
            raise RuntimeError("Could not create the --watch window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.focus_window(self.window)
        glfw.request_window_attention(self.window)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.scene = mujoco.MjvScene(self.model, maxgeom=4000)
        self.camera, self.option = mujoco.MjvCamera(), mujoco.MjvOption()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        center = self.offsets.mean(axis=0)
        self.camera.lookat[:] = [center[0] + 0.3, center[1], 0.1]
        self.camera.distance = 3.5 + 1.6 * columns
        self.camera.azimuth, self.camera.elevation = 150, -32
        self.cursor = None
        glfw.set_key_callback(
            self.window,
            lambda w, key, code, action, mods: (
                glfw.set_window_should_close(w, True)
                if key == glfw.KEY_ESCAPE and action == glfw.PRESS
                else None
            ),
        )
        glfw.set_cursor_pos_callback(self.window, self.drag)
        glfw.set_scroll_callback(
            self.window,
            lambda w, dx, dy: mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * dy, self.camera
            ),
        )

    def drag(self, window, x, y):
        glfw = self.glfw
        previous, self.cursor = self.cursor, (x, y)
        if previous is None:
            return
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or right):
            return
        _, height = glfw.get_window_size(window)
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if left else mujoco.mjtMouse.mjMOUSE_MOVE_H
        dx, dy = (x - previous[0]) / height, (y - previous[1]) / height
        mujoco.mjv_moveCamera(self.model, action, dx, dy, self.camera)

    def update(self, message):
        state = self.status[message["worker"]]
        if "done" in message:
            state["done"] = message.get("error") or f"done, {message['done']} episodes"
        elif "saving" in message:
            state["label"], state["step"] = "writing the episode file", 0
        elif "seed" in message:
            state.update({k: message[k] for k in ("seed", "tick", "label", "step", "plan")})
            if "goal" in message:
                self.goals[message["worker"]] = np.asarray(message["goal"])

    def open(self):
        return not self.glfw.window_should_close(self.window)

    def draw(self):
        glfw = self.glfw
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        for i, qpos in enumerate(self.states):
            if self.status[i]["seed"] is not None:
                self.data.qpos[self.index[i]] = qpos + self.shift @ self.offsets[i]
        mujoco.mj_forward(self.model, self.data)  # Kinematics only; never mj_step.
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        for goal, offset in zip(self.goals, self.offsets):
            if goal is not None:
                draw_goal(self.scene, goal, offset)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        lines = []
        for i, state in enumerate(self.status):
            if state["seed"] is None:
                lines.append(f"ARM {i + 1:2d}  starting")
            elif state["done"]:
                lines.append(f"ARM {i + 1:2d}  {state['done']}"[:70])
            else:
                prefix = f"{state['step']}/{state['plan']} " if state["step"] else ""
                lines.append(
                    f"ARM {i + 1:2d}  seed {state['seed']:<3d} {state['tick']:>5d}/{self.steps}  "
                    f"{prefix}{state['label']}"[:70]
                )
        # An overlay holds about 500 characters: split the list over two corners.
        half = (len(lines) + 1) // 2
        for chunk, corner in (
            (lines[:half], mujoco.mjtGridPos.mjGRID_TOPLEFT),
            (lines[half:], mujoco.mjtGridPos.mjGRID_TOPRIGHT),
        ):
            if chunk:
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    corner,
                    viewport,
                    "\n".join(chunk),
                    "",
                    self.context,
                )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            FINISHED
            if self.finished
            else "Drag: orbit | Right-drag: pan | Scroll: zoom | Esc: close window (collection continues)",
            "",
            self.context,
        )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()


class Wall:
    """A window tiling every worker's top camera with seed, progress and subtask."""

    def __init__(self, frames, target, steps, camera="context"):
        import glfw

        self.glfw, self.frames, self.steps = glfw, frames, steps
        self.finished = False
        self.status = [
            {"seed": None, "tick": 0, "label": "", "step": 0, "plan": 0, "done": None}
            for _ in frames
        ]
        height, width = IMAGE_SIZE
        # Up to five per row; prefer a grid with no empty cell (6 -> 3x2, 10 -> 5x2).
        self.columns = next(
            (c for c in range(min(len(frames), 5), 1, -1) if len(frames) % c == 0),
            min(len(frames), 5),
        )
        self.rows = math.ceil(len(frames) / self.columns)
        self.cell = (width, height + 26)
        if not glfw.init():
            raise RuntimeError(
                "Cannot open a display for --watch; drop the flag to record headless"
            )
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(
            self.columns * self.cell[0],
            self.rows * self.cell[1],
            f"Tangram-Bench | {len(frames)} arms collecting {target} | {camera} camera",
            None,
            None,
        )
        if not self.window:
            raise RuntimeError("Could not create the --watch window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.focus_window(self.window)
        glfw.request_window_attention(self.window)
        self.context = mujoco.MjrContext(
            mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"),
            mujoco.mjtFontScale.mjFONTSCALE_100,
        )
        glfw.set_key_callback(
            self.window,
            lambda w, key, code, action, mods: (
                glfw.set_window_should_close(w, True)
                if key == glfw.KEY_ESCAPE and action == glfw.PRESS
                else None
            ),
        )

    def update(self, message):
        state = self.status[message["worker"]]
        if "done" in message:
            state["done"] = message.get("error") or f"done, {message['done']} episodes"
        elif "saving" in message:
            state["label"], state["step"] = "writing the episode file", 0
        elif "seed" in message:
            state.update({k: message[k] for k in ("seed", "tick", "label", "step", "plan")})

    def open(self):
        return not self.glfw.window_should_close(self.window)

    def draw(self):
        glfw = self.glfw
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0.055, 0.065, 0.08, 1)
        w, h = IMAGE_SIZE[1], IMAGE_SIZE[0]
        for index, (frame, state) in enumerate(zip(self.frames, self.status)):
            column, row = index % self.columns, index // self.columns
            left = column * self.cell[0]
            bottom = height - (row + 1) * self.cell[1]
            cell = mujoco.MjrRect(left, bottom, self.cell[0], self.cell[1])
            # OpenGL rasters bottom-up; camera images come top-down.
            mujoco.mjr_drawPixels(
                np.ascontiguousarray(frame[::-1]).ravel(),
                None,
                mujoco.MjrRect(left, bottom, w, h),
                self.context,
            )
            if state["seed"] is None:
                title = f"ARM {index + 1}  starting"
            elif state["done"]:
                title = f"ARM {index + 1}  {state['done']}"
            else:
                title = f"ARM {index + 1}  seed {state['seed']}  {state['tick']}/{self.steps}"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                cell,
                title,
                "",
                self.context,
            )
            if state["label"] and not state["done"]:
                prefix = f"{state['step']}/{state['plan']}  " if state["step"] else ""
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    cell,
                    "\n".join(textwrap.wrap(prefix + state["label"], w // 7)[:2]),
                    "",
                    self.context,
                )
        if self.finished:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                mujoco.MjrRect(0, 0, width, height),
                FINISHED,
                "",
                self.context,
            )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()


def collect_parallel(seeds, args, out, watch):
    """Spread seeds round-robin over worker processes; the parent keeps the index."""
    from multiprocessing import shared_memory

    # Workers render offscreen; never let them open hidden GLFW windows on the desktop.
    os.environ.setdefault("MUJOCO_GL", "egl")
    seeds = list(seeds)
    workers = min(args.workers, len(seeds))
    context = mp.get_context("spawn")
    queue = context.Queue()
    blocks, frames, states, processes = [], [], [], []
    nq = Env(args.robot).model.nq
    try:
        for index in range(workers):
            block = shared_memory.SharedMemory(create=True, size=int(np.prod(IMAGE_SIZE)) * 3)
            blocks.append(block)
            frames.append(np.ndarray((*IMAGE_SIZE, 3), dtype=np.uint8, buffer=block.buf))
            frames[-1][:] = 0
            block = shared_memory.SharedMemory(create=True, size=nq * 8)
            blocks.append(block)
            states.append(np.ndarray(nq, dtype=np.float64, buffer=block.buf))
            states[-1][:] = 0
            process = context.Process(
                target=worker,
                args=(index, seeds[index::workers], args, out, queue, blocks[-2].name, block.name),
                daemon=True,
            )
            process.start()
            processes.append(process)
        wall = None
        try:
            if watch == "grid":
                wall = Wall(frames, args.target, args.steps, args.watch_camera)
            elif watch:
                wall = Fleet(states, args.robot, args.target, args.steps)
        except Exception as exc:  # No display, no GL: the recording must not depend on it.
            print(f"watch window unavailable ({exc}); recording headless", flush=True)
            wall = None
        kept, finished, errors = 0, 0, []
        reported = set()  # Workers whose done message arrived.
        recorded = {index: set() for index in range(workers)}
        while finished < workers:
            # Take everything queued since the last frame, so the window never lags.
            messages = []
            try:
                messages.append(queue.get(timeout=0.05))
                while len(messages) < 1000:
                    messages.append(queue.get_nowait())
            except Exception:
                pass
            for message in messages:
                if "row" in message:
                    append_index(out, message["row"])
                    recorded[message["worker"]].add(message["row"]["seed"])
                    done = sum(len(r) for r in recorded.values())
                    print(progress(message["row"], done, len(seeds)), flush=True)
                if "done" in message:
                    finished += 1
                    reported.add(message["worker"])
                    kept += message["done"]
                    if "error" in message:
                        errors.append(f"worker {message['worker']}: {message['error']}")
                if wall is not None:
                    wall.update(message)
            # A worker killed by the kernel (out of memory) or a hard exit never sends
            # its done message: notice its exit code instead of waiting forever.
            for index, process in enumerate(processes):
                if index not in reported and not process.is_alive():
                    finished += 1
                    reported.add(index)
                    missing = sorted(set(seeds[index::workers]) - recorded[index])
                    errors.append(
                        f"worker {index} exited with code {process.exitcode}; "
                        f"seeds not recorded: {missing}"
                    )
                    if wall is not None:
                        wall.update({"worker": index, "done": 0, "error": "exited"})
            if wall is not None:
                if wall.open():
                    wall.draw()
                else:
                    wall.close()
                    wall = None
                    print("watch window closed; collection continues", flush=True)
        for process in processes:
            process.join()
        if wall is not None:
            # Leave the final state on screen; the collection itself is over.
            wall.finished = True
            print(FINISHED, flush=True)
            try:
                while wall.open():
                    wall.draw()
            except KeyboardInterrupt:  # Everything is saved; Ctrl-C here is just "close".
                pass
            wall.close()
        for error in errors:
            print(error, flush=True)
        return kept, bool(errors)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for block in blocks:
            block.close()
            block.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--robot", choices=["panda", "piper"], default="panda")
    p.add_argument("--policy", type=Path, default=Path("examples/oracle.py"))
    p.add_argument("--split", choices=SPLITS, default="train")
    p.add_argument("--offset", type=int, default=None, help="First seed of the split to record")
    p.add_argument("--fps", type=int, default=10, help="Frame rate; must divide 50")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Horizon in control ticks")
    p.add_argument("--prompt", default=None, help="Override the per-figure task text")
    p.add_argument("--out", type=Path, help="Figure folder; default data/<target>")
    p.add_argument(
        "--round",
        help="Subfolder of --out for this launch; default the local date and time, "
        "YYYY-MM-DD-HH-MM-SS. Name an existing round to resume it (recorded seeds are skipped). "
        "Dated names sort in time order, which is how export and report pick the newest copy "
        "of a seed; a custom name sorts by its letters",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue the newest round of --out (or the one named by --round) with --episodes "
        "more seeds, starting after the last seed it lists; --offset overrides the start",
    )
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument(
        "--full-horizon", action="store_true", help="Do not stop early after a held success"
    )
    p.add_argument(
        "--goal-yaw", type=float, default=None, help="Force the goal yaw in degrees for every seed"
    )
    p.add_argument(
        "--source-yaw", type=float, default=None, help="Force the packed square's yaw in degrees"
    )
    p.add_argument("--workers", type=int, default=1, help="Arms recording at the same time")
    p.add_argument(
        "--watch",
        nargs="?",
        const="fleet",
        choices=("fleet", "grid"),
        help="Show every arm live: one 3D hall of tables (fleet) or a grid of cameras",
    )
    p.add_argument(
        "--watch-camera",
        choices=("context", *CAMERAS),
        default="context",
        help="Camera shown by --watch grid; context is the third-person view no policy sees",
    )
    args = p.parse_args(argv)
    if min(args.episodes, args.fps, args.steps, args.workers) < 1 or (args.offset or 0) < 0:
        p.error("episodes, fps, steps and workers must be positive; offset nonnegative")
    if CONTROL_HZ % args.fps:
        p.error(f"fps must divide the {CONTROL_HZ} Hz control rate")
    figure = args.out or Path("data") / args.target
    if args.resume:
        out = figure / args.round if args.round else latest_round(figure)
        if out is None or not out.is_dir():
            p.error(f"nothing to resume in {figure}")
    else:
        out = figure / (args.round or round_name())
    out.mkdir(parents=True, exist_ok=True)
    base = SPLITS[args.split]
    done = {seed for seed in recorded_seeds(out) if base <= seed < base + SPLIT_SIZE}
    offset = args.offset
    if offset is None:  # Resuming: the next seeds after the ones the round already lists.
        offset = max(done) + 1 - base if args.resume and done else 0
    print(json.dumps({"round": str(out), "recorded": len(done), "first_seed": base + offset}))
    args.policy = args.policy.resolve()
    seeds, skipped = [], 0
    for seed in range(base + offset, base + offset + args.episodes):
        if (out / f"episode-{seed}.npz").exists():
            print(json.dumps({"seed": seed, "skipped": "exists"}), flush=True)
            skipped += 1
        else:
            seeds.append(seed)
    started = time.perf_counter()
    failed = False
    if not seeds:
        kept = 0
    elif args.workers > 1 or args.watch:
        kept, failed = collect_parallel(seeds, args, out, args.watch)
    else:
        env = Env(args.robot, pixels=True)
        done = []

        def on_row(row):
            append_index(out, row)
            done.append(row["seed"])
            print(progress(row, len(done), len(seeds)), flush=True)

        kept = record_seeds(env, load_policy(args.policy), seeds, args, out, on_row=on_row)
        env.close()
    print(
        f"kept {kept}/{len(seeds)} episodes in {out} ({time.perf_counter() - started:.0f} s"
        + (f"; {skipped} already recorded" if skipped else "")
        + ")",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
