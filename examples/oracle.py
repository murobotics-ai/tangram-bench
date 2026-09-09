"""Privileged physical reference controller using the corpus solution certificates.

For manipulation calibration only, not the policy leaderboard. Every piece moves
through gripper contact. A private kinematic model supplies IK, never live state.
"""

from types import SimpleNamespace

import mujoco
import numpy as np

from env import make_model
from shapes import solution
from tangram import THICKNESS, VERTICES, knob_yaw
from teleop import Teleop, orientation_error, rotated


class Policy:
    access = "oracle"

    def __init__(self, robot, seed):
        self.seed = seed
        self.model = make_model(robot)
        self.data = mujoco.MjData(self.model)
        self.narm = 7 if robot == "panda" else 6
        self.high = 0.19 if robot == "panda" else 0.12
        self.order = [0, 1, 2, 6, 3, 4, 5]
        self.index = self.phase = self.elapsed = 0
        self.control = None
        self.source = None
        self.grasp_offset = 0
        self.retries = 0
        self.correction = np.zeros(3)
        self.setpoint = None
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        self.data.qpos[: self.narm + 2] = obs["qpos"]
        mujoco.mj_forward(self.model, self.data)
        if self.control is None:
            proxy = SimpleNamespace(
                model=self.model,
                data=[self.data],
                narm=self.narm,
                tcp=self.model.site("tcp").id,
                limits=self.model.actuator_ctrlrange.copy(),
                observe=lambda: [obs],
            )
            self.control = Teleop(proxy)
            self.setpoint = obs["tcp_pos"].copy()
            self.targets = solution(obs["target"], self.seed)
        if self.index >= 7:
            self.control.position = np.array([0.30, 0, 0.35])
            self.control.grip = 1
            return self.control.action()
        i = self.order[self.index]
        pose = obs["pieces"][i]
        if self.source is None:
            self.source = pose[:2].copy()
            w, x, y, z = pose[3:]
            piece_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            # Choose the closest equivalent orientation of the square knob.
            current_yaw = np.arctan2(obs["tcp_mat"][1, 0], obs["tcp_mat"][0, 0])
            desired = piece_yaw + knob_yaw(VERTICES[i])
            desired += round((current_yaw - desired) / (np.pi / 2)) * np.pi / 2
            self.grasp_offset = desired - piece_yaw
            self.yaw = desired
            original = self.data.qpos.copy()
            options = []
            for turn in range(4):
                candidate = desired + turn * np.pi / 2
                c, s = np.cos(candidate), np.sin(candidate)
                self.control.orientation = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]])
                self.control.position = np.r_[self.source, self.high]
                self.data.qpos[:] = original
                for attempt in range(12):
                    if attempt:
                        self.data.qpos[: self.narm] = self.rng.uniform(
                            self.model.actuator_ctrlrange[: self.narm, 0],
                            self.model.actuator_ctrlrange[: self.narm, 1],
                        )
                    for _ in range(12):
                        command = self.control.action()
                        self.data.qpos[: self.narm] = command[:-1]
                    cost = self.control.ik_error[0] + 0.2 * self.control.ik_error[1]
                    cost += 0.0001 * np.linalg.norm(command[:-1] - original[: self.narm])
                    options.append((cost, candidate, command[:-1].copy()))
                    if cost < 0.001:
                        break
            _, self.yaw, self.approach_joints = min(options, key=lambda item: item[0])
            self.grasp_offset = self.yaw - piece_yaw
            self.data.qpos[:] = original
        target = self.targets[i]
        destination = target[:2]
        low, high = THICKNESS + 0.020, self.high
        points = [
            (*self.source, high),
            (*self.source, low),
            (*self.source, low),
            (*self.source, high),
            (*destination, high),
            (*destination, low + 0.001),
            (*destination, low + 0.001),
            (*destination, high),
        ]
        w, x, y, z = target[3:]
        dest_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) + self.grasp_offset
        yaw = self.yaw if self.phase < 4 else dest_yaw
        c, s = np.cos(yaw), np.sin(yaw)
        desired_orientation = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]])
        angle = orientation_error(desired_orientation, obs["tcp_mat"])
        self.control.orientation = rotated(
            obs["tcp_mat"], angle * min(1, 0.08 / max(np.linalg.norm(angle), 1e-12))
        )
        desired_position = np.array(points[self.phase])
        residual = desired_position - obs["tcp_pos"]
        if np.linalg.norm(residual) < 0.03:
            self.correction = np.clip(self.correction + 0.1 * residual, -0.025, 0.025)
        delta = desired_position + self.correction - self.setpoint
        self.setpoint += delta * min(1, 0.0015 / max(np.linalg.norm(delta), 1e-12))
        offset = self.setpoint - obs["tcp_pos"]
        self.setpoint = obs["tcp_pos"] + offset * min(1, 0.03 / max(np.linalg.norm(offset), 1e-12))
        self.control.position = self.setpoint.copy()
        self.control.grip = 0 if self.phase in (2, 3, 4, 5) else 1
        action = self.control.action()
        if self.phase == 0 and (np.linalg.norm(residual) > 0.04 or np.linalg.norm(angle) > 0.15):
            action = np.r_[self.approach_joints, 1.0]
            self.setpoint = obs["tcp_pos"].copy()
        self.elapsed += 1
        close = np.linalg.norm(obs["tcp_pos"] - desired_position) < 0.003
        aligned = np.linalg.norm(orientation_error(desired_orientation, obs["tcp_mat"])) < 0.08
        wait = 60 if self.phase in (2, 6) else 10
        if close and aligned and self.elapsed >= wait:
            self.correction[:] = 0
            if self.phase == 3 and pose[2] < 0.06:
                self.retries += 1
                if self.retries > 2:
                    raise RuntimeError(f"Could not lift piece {i}")
                self.phase, self.elapsed, self.source = 0, 0, None
            else:
                self.phase, self.elapsed = self.phase + 1, 0
                if self.phase == 8:
                    self.index, self.phase, self.source, self.retries = self.index + 1, 0, None, 0
        elif self.elapsed > 600:
            raise RuntimeError(f"Unreachable or blocked waypoint: piece {i}, phase {self.phase}")
        action[:-1] = np.clip(
            action[:-1],
            self.model.actuator_ctrlrange[: self.narm, 0],
            self.model.actuator_ctrlrange[: self.narm, 1],
        )
        return action
