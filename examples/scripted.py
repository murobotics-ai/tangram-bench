"""Square-specific physical pick/place attempt, not a general Tangram solver.

The analytic square decomposition is public knowledge. All object motion comes
from contact: no teleportation, attachment constraints, or privileged simulator
access. This is a controller example; full-puzzle success is not guaranteed.
"""

import mujoco
import numpy as np

from env import make_model
from tangram import THICKNESS, square_solution


class Policy:
    def __init__(self, robot, seed):
        self.model = make_model(robot)  # Own kinematic model; never the evaluator's state.
        self.data = mujoco.MjData(self.model)
        self.narm = 7 if robot == "panda" else 6
        self.tcp = self.model.site("tcp").id
        self.limits = self.model.actuator_ctrlrange[: self.narm]
        self.order = [5, 3, 4, 2, 6, 0, 1]
        self.index, self.phase, self.elapsed = 0, 0, 0
        self.source = None

    def ik(self, obs, xyz, yaw):
        self.data.qpos[: self.narm + 2] = obs["qpos"]
        c, s = np.cos(yaw), np.sin(yaw)
        target = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]])
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(6):
            mujoco.mj_forward(self.model, self.data)
            current = self.data.site_xmat[self.tcp].reshape(3, 3)
            rotation_error = sum(np.cross(current[:, j], target[:, j]) for j in range(3)) * 0.5
            error = np.r_[xyz - self.data.site_xpos[self.tcp], rotation_error]
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tcp)
            jac = np.vstack((jp[:, : self.narm], jr[:, : self.narm]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(6), error)
            self.data.qpos[: self.narm] = np.clip(
                self.data.qpos[: self.narm] + np.clip(dq, -0.1, 0.1),
                self.limits[:, 0],
                self.limits[:, 1],
            )
        return self.data.qpos[: self.narm].copy()

    def act(self, obs):
        if obs.get("target", "square") != "square":
            raise ValueError("This legacy controller only supports the square target")
        if self.index >= 7:
            return np.r_[self.ik(obs, np.array([0.35, 0, 0.3]), 0), 1.0]
        i = self.order[self.index]
        xy, target_yaw = square_solution(obs["goal"])
        if self.source is None:
            pose = obs["pieces"][i]
            self.source = pose[:2].copy()
            w, x, y, z = pose[3:]
            self.yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        # Move above source, descend, pinch, lift, travel, lower, release, retreat.
        source, dest = self.source, xy[i]
        low, high = THICKNESS / 2 + 0.004, 0.15
        points = [
            (*source, high),
            (*source, low),
            (*source, low),
            (*source, high),
            (*dest, high),
            (*dest, low + 0.002),
            (*dest, low + 0.002),
            (*dest, high),
        ]
        yaw = self.yaw if self.phase < 4 else target_yaw
        # Large left triangle is easier to pinch along its narrower local x axis.
        yaw += np.pi / 2 if i == 1 else 0
        xyz = np.array(points[self.phase])
        grip = 0.0 if self.phase in (2, 3, 4, 5) else 1.0
        action = np.r_[self.ik(obs, xyz, yaw), grip]
        self.elapsed += 1
        close = np.linalg.norm(obs["tcp_pos"] - xyz) < 0.008
        # Fixed waits give the fingers time to close/release. Timeout prevents deadlock.
        if (close and self.elapsed >= (40 if self.phase in (2, 6) else 25)) or self.elapsed >= 150:
            self.phase, self.elapsed = self.phase + 1, 0
            if self.phase == 8:
                self.index, self.phase, self.source = self.index + 1, 0, None
        return action
