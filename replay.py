"""Read-only episode replay from the recorded model and states; never rerun a policy."""

from pathlib import Path

import mujoco
import numpy as np

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
        self.robot, self.prompt = self.result["robot"], self.result["prompt"]
        self.target = self.row["target"]
        self.outline = self.trace["goal"][0]
        self.narm = 7 if self.robot == "panda" else 6
        self.frames = len(self.trace["time"])
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

    def text(self):
        return (
            f"REPLAY | seed {self.row['seed']} | {self.row['status']}\n"
            f"Step {self.steps}/{self.frames - 1} | {self.data[0].time:.2f} s | "
            f"Final success: {self.row['success']}\n"
            "Space: pause | Left/right: seek 1 s | Home/end: first/last"
        )
