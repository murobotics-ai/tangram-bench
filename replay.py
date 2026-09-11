"""Read-only replay of evaluation episodes and recorded demonstrations; never reruns a policy."""

from pathlib import Path

import mujoco
import numpy as np

from benchmark import CONTROL_SECONDS
from env import make_model
from tools.results import verify


class Replay:
    def __init__(self, path, episode=0):
        path = Path(path)
        self.result = verify(path)
        if not 0 <= episode < len(self.result["episodes"]):
            raise ValueError("Episode index is outside the result")
        self.row = self.result["episodes"][episode]
        with np.load(path.parent / self.row["trajectory"], allow_pickle=False) as trace:
            self.trace = {key: trace[key].copy() for key in trace.files}
        self.model = mujoco.MjModel.from_binary_path(str(path.parent / self.result["model"]))
        self.data = [mujoco.MjData(self.model)]
        self.robot, self.target = self.result["robot"], self.row["target"]
        self.prompt = str(self.trace["prompt"])
        self.subtasks = [str(t) for t in self.trace.get("subtask", [])]
        self.plan = [str(t) for t in self.trace.get("plan", [])]
        self.step_index = [int(t) for t in self.trace.get("step", [])]
        self.outline = self.trace["goal"][0]
        self.narm = 7 if self.robot == "panda" else 6
        self.frames = len(self.trace["time"])
        self.frame_seconds = CONTROL_SECONDS
        self.steps = 0
        self.seek(0)

    def seek(self, frame):
        self.steps = int(np.clip(frame, 0, self.frames - 1))
        d = self.data[0]
        d.qpos[: self.narm + 2] = self.trace["qpos"][self.steps]
        d.qvel[: self.narm + 2] = self.trace["qvel"][self.steps]
        for i in range(7):
            joint = self.model.joint(f"piece{i}")
            q, v = joint.qposadr[0], joint.dofadr[0]
            d.qpos[q : q + 7] = self.trace["pieces"][self.steps, i]
            d.qvel[v : v + 6] = self.trace["piece_velocities"][self.steps, i]
        if self.steps:
            d.ctrl[:] = self.trace["controls"][self.steps - 1]
        d.time = self.trace["time"][self.steps]
        mujoco.mj_forward(self.model, d)

    def subtask(self):
        """Language annotation for the current step, if the policy recorded one."""
        index = min(max(self.steps - 1, 0), len(self.subtasks) - 1)
        return self.subtasks[index] if self.subtasks else ""

    def label(self):
        return subtask_label(self.subtask(), self.step_index, self.plan, max(self.steps - 1, 0))

    def text(self):
        return (
            f"REPLAY | seed {self.row['seed']} | {self.row['status']}\n"
            f"Step {self.steps}/{self.frames - 1} | {self.data[0].time:.2f} s | "
            f"Final success: {self.row['success']}\n"
            "Space: pause | Left/right: seek 1 s | Home/end: first/last"
        )


def subtask_label(text, step_index, plan, frame):
    """'SUBTASK 3/7: ...' when the demonstrator numbered its plan, else 'SUBTASK: ...'."""
    if not text:
        return ""
    step = step_index[min(frame, len(step_index) - 1)] if step_index else 0
    if step and plan:
        return f"SUBTASK {step}/{len(plan)}: {text}"
    return f"SUBTASK: {text}"


class Demo:
    """Replay a demonstration recorded by tools/collect.py or the teleoperation recorder."""

    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as data:
            self.trace = {key: data[key] for key in data.files}
        self.robot, self.target = str(self.trace["robot"]), str(self.trace["target"])
        self.prompt = str(self.trace["prompt"])
        self.subtasks = [str(t) for t in self.trace.get("subtask", [])]
        self.plan = [str(t) for t in self.trace.get("plan", [])]
        self.step_index = [int(t) for t in self.trace.get("step", [])]
        self.success, self.seed = bool(self.trace["success"]), int(self.trace["seed"])
        self.model = make_model(self.robot)
        self.data = [mujoco.MjData(self.model)]
        self.outline = self.trace["goal"]
        self.narm = 7 if self.robot == "panda" else 6
        self.frames = len(self.trace["time"])
        self.frame_seconds = 1 / int(self.trace["fps"])
        self.steps = 0
        self.seek(0)

    def seek(self, frame):
        self.steps = int(np.clip(frame, 0, self.frames - 1))
        d = self.data[0]
        d.qpos[: self.narm + 2] = self.trace["state"][self.steps]
        for i in range(7):
            q = self.model.joint(f"piece{i}").qposadr[0]
            d.qpos[q : q + 7] = self.trace["pieces"][self.steps, i]
        d.time = float(self.trace["time"][self.steps])
        mujoco.mj_forward(self.model, d)

    def subtask(self):
        return self.subtasks[self.steps] if self.subtasks else ""

    def label(self):
        return subtask_label(self.subtask(), self.step_index, self.plan, self.steps)

    def text(self):
        return (
            f"DEMO | seed {self.seed} | {'solved' if self.success else 'not solved'}\n"
            f"Frame {self.steps}/{self.frames - 1} | {self.data[0].time:.1f} s\n"
            "Space: pause | Left/right: seek 1 s | Home/end: first/last"
        )
