"""Keyboard TCP targets -> damped IK -> existing joint actuators. No policy or recorder."""

import mujoco
import numpy as np

from env import DT, SUBSTEPS

CONTROL_DT = DT * SUBSTEPS


def orientation_error(target, current):
    """Shortest rotation vector in world coordinates, including 180-degree errors."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (target @ current.T).ravel())
    if quat[0] < 0:
        quat = -quat
    length = np.linalg.norm(quat[1:])
    return quat[1:] * (2 * np.arctan2(length, quat[0]) / max(length, 1e-12))


def rotated(current, vector):
    angle = np.linalg.norm(vector)
    quat = np.r_[np.cos(angle / 2), np.asarray(vector) * np.sin(angle / 2) / max(angle, 1e-12)]
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3) @ current


class Teleop:
    def __init__(self, env):
        self.env = env
        self.scratch = mujoco.MjData(env.model)
        self.jp = np.zeros((3, env.model.nv))
        self.jr = np.zeros_like(self.jp)
        obs = env.observe()[0]
        self.position = obs["tcp_pos"].copy()
        self.orientation = obs["tcp_mat"].copy()
        self.grip = 1.0
        self.paused = False
        self.error = self.angle_error = 0.0
        self.ik_error = np.zeros(2)
        self.singular = False
        self.posture = obs["qpos"][: env.narm].copy()

    def move(self, translation, rotation, grip=0.0, slow=False):
        scale = CONTROL_DT * (0.2 if slow else 1.0)
        self.position += np.asarray(translation) * 0.08 * scale
        # Limit target wind-up against contacts or unreachable poses.
        actual = self.env.data[0].site_xpos[self.env.tcp]
        offset = self.position - actual
        self.position = actual + offset * min(1, 0.04 / max(np.linalg.norm(offset), 1e-12))
        self.orientation = rotated(self.orientation, np.asarray(rotation) * 0.6 * scale)
        current = self.env.data[0].site_xmat[self.env.tcp].reshape(3, 3)
        offset = orientation_error(self.orientation, current)
        self.orientation = rotated(
            current, offset * min(1, 0.35 / max(np.linalg.norm(offset), 1e-12))
        )
        self.grip = float(np.clip(self.grip + grip * scale, 0, 1))

    def residual(self):
        env, d = self.env, self.scratch
        mujoco.mj_kinematics(env.model, d)
        mujoco.mj_comPos(env.model, d)
        return np.r_[
            self.position - d.site_xpos[env.tcp],
            0.2 * orientation_error(self.orientation, d.site_xmat[env.tcp].reshape(3, 3)),
        ]

    def action(self):
        env, d = self.env, self.scratch
        d.qpos[:] = env.data[0].qpos
        error = self.residual()
        self.singular = False
        for _ in range(12):
            mujoco.mj_jacSite(env.model, d, self.jp, self.jr, env.tcp)
            jac = np.vstack((self.jp[:, : env.narm], 0.2 * self.jr[:, : env.narm]))
            u, values, vt = np.linalg.svd(jac, full_matrices=True)
            self.singular |= values[-1] < 0.01
            damping = 0.002 + 0.03 * max(0, 1 - values[-1] / 0.05) ** 2
            delta = vt[:6].T @ ((values / (values**2 + damping**2)) * (u.T @ error))
            if env.narm > 6:
                # Exact redundant subspace: prefer the reset posture without a damped projector's task leakage.
                null = vt[6:].T
                delta += 0.15 * null @ (null.T @ (self.posture - d.qpos[: env.narm]))
            delta *= min(1, 0.1 / max(np.max(abs(delta)), 1e-12))
            previous = d.qpos[: env.narm].copy()
            # Clipping at joint limits can spoil a Newton step; reject residual increases.
            for fraction in (1.0, 0.5, 0.25):
                d.qpos[: env.narm] = np.clip(
                    previous + fraction * delta, env.limits[:-1, 0], env.limits[:-1, 1]
                )
                candidate = self.residual()
                if np.linalg.norm(candidate) <= max(np.linalg.norm(error), 1e-4):
                    error = candidate
                    break
            else:
                d.qpos[: env.narm] = previous
                error = self.residual()
                break
        error = self.residual()
        self.ik_error = np.array([np.linalg.norm(error[:3]), np.linalg.norm(error[3:]) / 0.2])
        return np.r_[d.qpos[: env.narm], self.grip]

    def step(self, translation, rotation, grip=0.0, slow=False):
        if self.paused:
            return
        self.move(translation, rotation, grip, slow)
        obs = self.env.step([self.action()])[0]
        self.error = float(np.linalg.norm(self.position - obs["tcp_pos"]))
        self.angle_error = float(
            np.linalg.norm(orientation_error(self.orientation, obs["tcp_mat"]))
        )

    def text(self):
        state = "PAUSED" if self.paused else "TELEOP"
        status = (
            "OK"
            if self.ik_error[0] < 0.001 and self.ik_error[1] < np.deg2rad(1)
            else "NOT CONVERGED"
        )
        return (
            f"{state} | Gripper {self.grip:.0%} open | IK {status}"
            + (" / near singular" if self.singular else "")
            + "\n"
            f"IK: {self.ik_error[0] * 1000:.1f} mm / {np.rad2deg(self.ik_error[1]):.1f} deg\n"
            f"Tracking: {self.error * 1000:.1f} mm / {np.rad2deg(self.angle_error):.1f} deg\n"
            "W/S: X  A/D: Y  Q/E: up/down  (world axes)\n"
            "Z/X: roll  T/G: pitch  C/V: yaw\n"
            "R: open  F: close (hold)  O: reset  |  Shift: fine\n"
            "P: pause  Esc: exit"
        )
