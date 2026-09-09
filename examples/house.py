"""House-specific Panda reference controller; all motion uses joint actions/contact.

Uses the published house certificate, so it is an oracle calibration policy,
not a learned/generalization baseline. Does not change the evaluator or physics.
Validated demonstration: Panda, development seed 100076, 12000 control steps.
Other seeds and the standard 3000-step horizon are not validated by this demo.
"""

from types import SimpleNamespace

import mujoco
import numpy as np

from env import HOME, make_model
from shapes import solution
from tangram import VERTICES, knob_yaw, score
from teleop import Teleop, orientation_error, rotated


def matrix(quat):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quat, dtype=float))
    return result.reshape(3, 3)


def downward(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


class Policy:
    access = "oracle"

    def __init__(self, robot, seed):
        if robot != "panda":
            raise ValueError("The house reference controller currently supports Panda only")
        self.model = make_model(robot)
        self.data = mujoco.MjData(self.model)
        self.targets = solution("house", seed)
        self.order = [0, 1, 2, 6, 3, 4, 5]
        self.index = 0
        self.phase = "start"
        self.ticks = self.stable = 0
        self.control = None
        self.bias = np.zeros(3)
        self.rng = np.random.default_rng(seed)
        self.high = 0.22
        self.last_action = None

    def choose_grasp(self, obs):
        pose = obs["pieces"][self.piece]
        yaw = np.arctan2(matrix(pose[3:])[1, 0], matrix(pose[3:])[0, 0])
        yaw += knob_yaw(VERTICES[self.piece])
        options = []
        original = self.data.qpos.copy()
        # The square knob admits four equivalent grasps. Check source, transit
        # and destination IK so a convenient pickup does not exhaust wrist travel.
        for turn in range(4):
            orient = downward(yaw + turn * np.pi / 2)
            for attempt in range(4):
                self.data.qpos[:] = original
                if attempt:
                    self.data.qpos[:7] = np.array(HOME["panda"]) + self.rng.uniform(-0.7, 0.7, 7)
                    self.data.qpos[:7] = np.clip(
                        self.data.qpos[:7], self.limits[:, 0], self.limits[:, 1]
                    )
                self.control.position = np.r_[pose[:2], self.high]
                self.control.orientation = orient
                for _ in range(16):
                    cmd = self.control.action()
                    self.data.qpos[:7] = cmd[:7]
                err = self.control.ik_error
                margin = np.minimum(cmd[:7] - self.limits[:, 0], self.limits[:, 1] - cmd[:7])
                cost = 1000 * max(0, err[0] - 0.001) + 100 * max(0, err[1] - 0.01)
                cost += np.linalg.norm(cmd[:7] - original[:7]) + 0.02 * np.sum(
                    1 / np.maximum(margin, 0.01)
                )
                approach = cmd[:7].copy()
                target = self.targets[self.piece]
                end_R = matrix(target[3:]) @ matrix(pose[3:]).T @ orient
                postures = []
                for xyz, rot in [
                    (np.r_[pose[:2], 0.025], orient),
                    (np.r_[pose[:2], self.high], orient),
                    (np.array([0.5, 0.0, self.high]), orient),
                    (np.array([0.5, 0.0, self.high]), end_R),
                    (np.r_[target[:2], self.high], end_R),
                    (np.r_[target[:2], 0.03], end_R),
                ]:
                    self.control.position, self.control.orientation = xyz, rot
                    for _ in range(20):
                        cmd = self.control.action()
                        self.data.qpos[:7] = cmd[:7]
                    err = self.control.ik_error
                    margin = np.minimum(cmd[:7] - self.limits[:, 0], self.limits[:, 1] - cmd[:7])
                    cost += 1000 * max(0, err[0] - 0.001) + 100 * max(0, err[1] - 0.01)
                    cost += 0.1 * np.sum(1 / np.maximum(margin, 0.01))
                    postures.append(cmd[:7].copy())
                options.append((cost, orient.copy(), approach, postures))
        _, self.grasp_orientation, self.approach, self.postures = min(options, key=lambda x: x[0])
        self.data.qpos[:] = original
        self.source = pose[:2].copy()

    def transition(self, phase, obs):
        self.phase, self.ticks, self.stable = phase, 0, 0
        if phase != "release":
            self.bias[:] = 0
        self.setpoint = obs["tcp_pos"].copy()

    def act(self, obs):
        if obs["target"] != "house":
            raise ValueError("This controller is specific to the house silhouette")
        self.data.qpos[:9] = obs["qpos"]
        mujoco.mj_forward(self.model, self.data)
        if self.control is None:
            self.limits = self.model.actuator_ctrlrange[:7]
            proxy = SimpleNamespace(
                model=self.model,
                data=[self.data],
                narm=7,
                tcp=self.model.site("tcp").id,
                limits=self.model.actuator_ctrlrange,
                observe=lambda: [obs],
            )
            self.control = Teleop(proxy)
            self.setpoint = obs["tcp_pos"].copy()
        if self.phase == "start":
            if self.index == 7:
                self.transition("done", obs)
            else:
                self.piece = self.order[self.index]
                self.choose_grasp(obs)
                self.transition("approach", obs)
        if self.phase == "done":
            # Hold the compensated command, not a continuously drifting measured pose.
            return np.r_[self.last_action[:7], 1.0]
        # Stop correcting the grasp once the observed assembly is already solved,
        # including a piece that has settled out of the fingers.
        if (
            self.phase in ("transfer", "lower")
            and score(obs["pieces"], obs["piece_velocities"], obs["goal"])["success"]
        ):
            self.release_joints = self.last_action[:7].copy()
            self.transition("release", obs)
        if self.phase == "release":
            self.ticks += 1
            action = np.r_[self.release_joints, 1.0]
            if self.ticks >= 50:
                self.release_position = obs["tcp_pos"].copy()
                self.release_R = obs["tcp_mat"].copy()
                self.transition("retreat", obs)
            self.last_action = action.copy()
            return action
        posture_index = {
            "descend": 0,
            "close": 0,
            "lift": 1,
            "transit": 3,
            "transfer": 4,
            "lower": 5,
            "retreat": 4,
        }.get(self.phase)
        self.control.posture = np.array(HOME["panda"])
        if np.linalg.norm(self.source) < 0.29 and posture_index is not None:
            self.control.posture = self.postures[posture_index]
        pose = obs["pieces"][self.piece]
        target = self.targets[self.piece]
        R = self.grasp_orientation
        grip = 0.0 if self.phase in ("close", "lift", "transit", "transfer", "lower") else 1.0
        if self.phase in ("approach", "descend", "close", "lift"):
            position = np.r_[
                self.source, self.high if self.phase in ("approach", "lift") else 0.025
            ]
        elif self.phase in ("transit", "transfer", "lower"):
            # Measure the grasp transform again each tick to correct small slips.
            piece_R = matrix(pose[3:])
            relative_R = obs["tcp_mat"].T @ piece_R
            relative_p = obs["tcp_mat"].T @ (pose[:3] - obs["tcp_pos"])
            R = matrix(target[3:]) @ relative_R.T
            desired_piece = target[:3].copy()
            if self.phase == "transit":
                desired_piece[:2] = [0.5, 0.0]
            if self.phase != "lower":
                desired_piece[2] = self.high - 0.025
            else:
                # Release just above the table; pressing the slab down while
                # constrained by the gripper makes it slide when the jaws open.
                desired_piece[2] += 0.008
            position = desired_piece - R @ relative_p
            self.release_position, self.release_R = position.copy(), R.copy()
        else:
            position, R = self.release_position.copy(), self.release_R
            if self.phase == "retreat":
                position[2] += 0.15
        residual = position - obs["tcp_pos"]
        if np.linalg.norm(residual) < 0.04:
            self.bias = np.clip(self.bias + 0.06 * residual, -0.025, 0.025)
        desired = position + self.bias
        delta = desired - self.setpoint
        speed = 0.0012 if self.phase in ("lift", "transit", "transfer", "lower") else 0.002
        self.setpoint += delta * min(1.0, speed / max(np.linalg.norm(delta), 1e-12))
        offset = self.setpoint - obs["tcp_pos"]
        self.setpoint = obs["tcp_pos"] + offset * min(
            1.0, 0.03 / max(np.linalg.norm(offset), 1e-12)
        )
        self.control.position = self.setpoint.copy()
        error = orientation_error(R, obs["tcp_mat"])
        compensated = error
        self.control.orientation = rotated(
            obs["tcp_mat"], compensated * min(1.0, 0.08 / max(np.linalg.norm(compensated), 1e-12))
        )
        self.control.grip = grip
        action = self.control.action()
        if self.phase == "approach" and self.ticks < 200:
            action = np.r_[self.approach, 1.0]
            self.setpoint = obs["tcp_pos"].copy()
        self.ticks += 1
        near = np.linalg.norm(residual) < (0.0015 if self.phase == "lower" else 0.006)
        aligned = np.linalg.norm(error) < (0.02 if self.phase == "lower" else 0.06)
        self.stable = self.stable + 1 if near and aligned else 0
        wait = 60 if self.phase == "close" else 8
        if self.stable >= wait:
            if self.phase == "lower":
                self.release_joints = action[:7].copy()
            phases = [
                "approach",
                "descend",
                "close",
                "lift",
                "transit",
                "transfer",
                "lower",
                "release",
                "retreat",
            ]
            if self.phase == "lift" and pose[2] < 0.08:
                raise RuntimeError(f"Grasp failed for piece {self.piece}")
            if self.phase == "retreat":
                self.index += 1
                self.transition("start", obs)
            else:
                self.transition(phases[phases.index(self.phase) + 1], obs)
        elif self.ticks > 1000:
            raise RuntimeError(f"Blocked at piece {self.piece}, phase {self.phase}")
        action[:7] = np.clip(action[:7], self.limits[:, 0], self.limits[:, 1])
        self.last_action = action.copy()
        return action
