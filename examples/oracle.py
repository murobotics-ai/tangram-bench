"""Privileged reference controller: any silhouette, from its certificate, via contact.

Uses the published solution certificate, so it is an oracle demonstrator
(access = "oracle"), not a learned baseline. Every piece moves through gripper
contact using the ordinary joint-action interface; the evaluator and physics
are untouched. It picks among the four equivalent knob grasps and two carry
heights by checking IK at pickup, transit and placement, hovers above the target,
descends slowly, releases once the observed piece rests within tolerance, and
re-grasps a piece it loses. Panda only. Reliability across seeds is measured
(`eval.py --policy examples/oracle.py`, `tools/collect.py`) and reported in the
README, never assumed.
"""

from types import SimpleNamespace

import mujoco
import numpy as np

from env import HOME, make_model
from shapes import solution
from tangram import NAMES, THICKNESS, VERTICES, goal_transform, knob_yaw, rotation, score
from teleop import Teleop, orientation_error, rotated


def matrix(quat):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quat, dtype=float))
    return result.reshape(3, 3)


def downward(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def region(xy, outline, yaw):
    """Where a point lies within a silhouette, in the silhouette's own frame: 'top-left' etc."""
    center = (outline.min(axis=0) + outline.max(axis=0)) / 2
    local = rotation(-yaw) @ (np.asarray(xy) - center)
    half = np.ptp(outline @ rotation(-yaw).T, axis=0) / 2
    x, y = local / np.maximum(half, 1e-9)
    row = "top" if y > 0.25 else "bottom" if y < -0.25 else ""
    column = "right" if x > 0.25 else "left" if x < -0.25 else ""
    return "-".join(w for w in (row, column) if w) or "center"


def placement_error(pose, target):
    """Planar distance in metres and yaw difference in radians between two piece poses."""
    yaw = np.arctan2(matrix(pose[3:])[1, 0], matrix(pose[3:])[0, 0])
    goal_yaw = np.arctan2(matrix(target[3:])[1, 0], matrix(target[3:])[0, 0])
    angle = (yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(pose[:2] - target[:2])), float(abs(angle))


HEIGHTS = (0.22, 0.14)  # Candidate TCP carry heights; the lower one serves poses near the base.
HOVER = 0.03  # Piece height above the table where the final descent starts.
CARRY_SPEED = 0.0012  # Setpoint travel per tick while a piece is held.
LOWER_SPEED = 0.0006  # Slow final descent: fast descents creep the slab out of the fingers.
RELEASE = 0.004  # Piece height above the table at release; higher drops land on neighbours.
PHASES = (
    "approach",
    "descend",
    "close",
    "lift",
    "transit",
    "transfer",
    "hover",
    "lower",
    "release",
    "retreat",
)


class Policy:
    access = "oracle"

    def __init__(self, robot, seed):
        if robot != "panda":
            raise ValueError("The reference controller currently supports Panda only")
        self.seed = seed
        self.model = make_model(robot)
        self.data = mujoco.MjData(self.model)
        self.targets = None
        self.order = [0, 1, 2, 6, 3, 4, 5]
        self.index = 0
        self.phase = "start"
        self.ticks = self.stable = 0
        self.control = None
        self.bias = np.zeros(3)
        self.rng = np.random.default_rng(seed)
        self.high = HEIGHTS[0]
        self.last_action = None
        # Language annotations for demonstrations: a numbered plan (one step per
        # piece), the 1-based step in progress and the fine-grained subtask sentence.
        self.plan, self.step = [], 0
        self.subtask = "Look at the silhouette on the table and plan the assembly."
        self.retries = 0  # Re-grasp attempts for the current piece.
        self.descents = 0  # Repeated final descents for the current grasp.
        self.rate = 0.0  # Current setpoint speed, ramped after each transition.
        self.retries_total = 0
        self.regrasp = False  # A retreat that returns to the same piece.

    def choose_grasp(self, obs):
        pose = obs["pieces"][self.piece]
        yaw = np.arctan2(matrix(pose[3:])[1, 0], matrix(pose[3:])[0, 0])
        yaw += knob_yaw(VERTICES[self.piece])
        options = []
        original = self.data.qpos.copy()
        # The square knob admits four equivalent grasps. Check source, transit
        # and destination IK at each carry height so a convenient pickup does not
        # exhaust wrist travel or push the elbow against its limit near the base.
        for high, turn in [(h, t) for h in HEIGHTS for t in range(4)]:
            orient = downward(yaw + turn * np.pi / 2)
            for attempt in range(4):
                self.data.qpos[:] = original
                if attempt:
                    self.data.qpos[:7] = np.array(HOME["panda"]) + self.rng.uniform(-0.7, 0.7, 7)
                    self.data.qpos[:7] = np.clip(
                        self.data.qpos[:7], self.limits[:, 0], self.limits[:, 1]
                    )
                self.control.position = np.r_[pose[:2], high]
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
                    (np.r_[pose[:2], high], orient),
                    (np.array([0.5, 0.0, high]), orient),
                    (np.array([0.5, 0.0, high]), end_R),
                    (np.r_[target[:2], high], end_R),
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
                options.append((cost, high, orient.copy(), approach, postures))
        _, self.high, self.grasp_orientation, self.approach, self.postures = min(
            options, key=lambda x: x[0]
        )
        self.data.qpos[:] = original
        self.source = pose[:2].copy()

    def describe(self, obs):
        """One sentence saying what the controller is doing now, for language annotations."""
        if self.phase == "done":
            return "All seven pieces are placed; hold still."
        name, figure = NAMES[self.piece], obs["target"]
        where = region(self.targets[self.piece][:2], obs["goal"], self.yaw)
        if self.regrasp and self.phase in ("release", "retreat"):
            return f"The {name} slipped; let go, back away and pick it up again."
        if self.phase in ("approach", "descend", "close"):
            return f"Reach for the {name} and grasp its knob from above."
        if self.phase == "lift":
            return f"Lift the {name} off the table."
        if self.phase in ("transit", "transfer", "hover"):
            return f"Carry the {name} to the {where} of the {figure} and align it."
        if self.phase == "lower":
            return f"Lower the {name} into place at the {where} of the {figure}."
        return f"Release the {name} and back away."

    def transition(self, phase, obs):
        # Keep the integrated servo bias through release and the hover/lower pair,
        # where it holds the planar alignment reached while hovering.
        if phase not in ("release", "lower") and not (phase == "hover" and self.phase == "lower"):
            self.bias[:] = 0
        self.phase, self.ticks, self.stable = phase, 0, 0
        self.setpoint = obs["tcp_pos"].copy()
        self.rate = 0.0

    def recover(self, obs, reason):
        """Let go, back off and pick the same piece again from where it lies now."""
        self.retries += 1
        self.retries_total += 1
        if self.retries > 3:
            raise RuntimeError(f"{reason}: piece {self.piece} after {self.retries - 1} retries")
        self.regrasp = True
        self.release_joints = self.last_action[:7].copy()
        self.release_position, self.release_R = obs["tcp_pos"].copy(), obs["tcp_mat"].copy()
        self.transition("release", obs)

    def act(self, obs):
        if self.targets is None:
            self.targets = solution(obs["target"], self.seed)
            self.yaw = goal_transform(self.seed, obs["target"])[1]
            self.plan = [
                f"Place the {NAMES[i]} at the {region(self.targets[i][:2], obs['goal'], self.yaw)} "
                f"of the {obs['target']}."
                for i in self.order
            ]
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
        self.step = min(self.index + 1, len(self.order))
        self.subtask = self.describe(obs)
        if self.phase == "done":
            # Hold the compensated command, not a continuously drifting measured pose.
            return np.r_[self.last_action[:7], 1.0]

        # Stop correcting the grasp once the piece is already resting at its target,
        # including a piece that has settled out of the fingers, or once the whole
        # assembly is solved.
        if self.phase in ("transfer", "hover", "lower"):
            pose = obs["pieces"][self.piece]
            distance, angle = placement_error(pose, self.targets[self.piece])
            resting = pose[2] < THICKNESS / 2 + 0.002
            if (resting and distance < 0.0025 and angle < np.deg2rad(2.5)) or score(
                obs["pieces"], obs["piece_velocities"], obs["goal"]
            )["success"]:
                self.release_joints = self.last_action[:7].copy()
                self.transition("release", obs)
            elif resting and self.ticks > 20:
                if self.phase == "lower" and self.descents < 3:
                    # Touched down off target while still held: lift and descend again.
                    self.descents += 1
                    self.transition("hover", obs)
                else:
                    # The slab slipped out or was pushed onto the table off target.
                    self.recover(obs, "Piece lost before placement")
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
            "hover": 5,
            "lower": 5,
            "retreat": 4,
        }.get(self.phase)
        self.control.posture = np.array(HOME["panda"])
        if np.linalg.norm(self.source) < 0.29 and posture_index is not None:
            self.control.posture = self.postures[posture_index]
        pose = obs["pieces"][self.piece]
        target = self.targets[self.piece]
        R = self.grasp_orientation
        carrying = self.phase in ("lift", "transit", "transfer", "hover", "lower")
        grip = 0.0 if self.phase == "close" or carrying else 1.0
        if self.phase in ("approach", "descend", "close", "lift"):
            position = np.r_[
                self.source, self.high if self.phase in ("approach", "lift") else 0.025
            ]
        elif self.phase in ("transit", "transfer", "hover", "lower"):
            # Measure the grasp transform again each tick to correct small slips.
            piece_R = matrix(pose[3:])
            relative_R = obs["tcp_mat"].T @ piece_R
            relative_p = obs["tcp_mat"].T @ (pose[:3] - obs["tcp_pos"])
            R = matrix(target[3:]) @ relative_R.T
            desired_piece = target[:3].copy()
            if self.phase == "transit":
                desired_piece[:2] = [0.5, 0.0]
            if self.phase in ("transit", "transfer"):
                desired_piece[2] = self.high - 0.025
            elif self.phase == "hover":
                # Align in the plane first, then descend straight down.
                desired_piece[2] += HOVER
            else:
                # Release just above the table; pressing the slab down while
                # constrained by the gripper makes it slide when the jaws open.
                desired_piece[2] += RELEASE
            position = desired_piece - R @ relative_p
            self.release_position, self.release_R = position.copy(), R.copy()
        else:
            position, R = self.release_position.copy(), self.release_R
            if self.phase == "retreat":
                # Back off to the carry height whose reachability was checked.
                position[2] = max(position[2], self.high)
        residual = position - obs["tcp_pos"]
        speed = 0.002
        if self.phase in ("lift", "transit", "transfer", "hover"):
            speed = CARRY_SPEED
        elif self.phase == "lower":
            speed = LOWER_SPEED
        # Integrate the servo's steady-state error only once the ramped setpoint has
        # caught up with the request; otherwise a slow descent winds the bias up.
        if (
            np.linalg.norm(residual) < 0.04
            and np.linalg.norm(position + self.bias - self.setpoint) < 2 * speed
        ):
            self.bias = np.clip(self.bias + 0.06 * residual, -0.025, 0.025)
        desired = position + self.bias
        delta = desired - self.setpoint
        # Ramp the setpoint speed up over about half a second: a step to full speed
        # jerks the arm and pivots the slab about the finger contact.
        self.rate = min(speed, self.rate + speed / 25)
        self.setpoint += delta * min(1.0, self.rate / max(np.linalg.norm(delta), 1e-12))
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
        tolerance = {"lower": (0.0015, 0.02), "hover": (0.002, 0.03)}.get(self.phase, (0.006, 0.06))
        near = np.linalg.norm(residual) < tolerance[0]
        aligned = np.linalg.norm(error) < tolerance[1]
        self.stable = self.stable + 1 if near and aligned else 0
        wait = 60 if self.phase == "close" else 8
        if self.stable >= wait:
            if self.phase == "lower":
                self.release_joints = action[:7].copy()
            if self.phase == "lift" and pose[2] < 0.08:
                self.recover(obs, "Grasp failed")
            elif self.phase == "retreat":
                settled = pose[2] < THICKNESS / 2 + 0.002
                if self.regrasp:
                    pass  # Pick the same piece up again from where it lies.
                elif settled:
                    self.index += 1
                    self.retries = 0
                else:
                    # Landed on a neighbour's edge: pick it up and place it again.
                    self.retries += 1
                    self.retries_total += 1
                    if self.retries > 3:
                        raise RuntimeError(f"Piece {self.piece} rests on a neighbour")
                self.regrasp, self.descents = False, 0
                self.transition("start", obs)
            else:
                self.transition(PHASES[PHASES.index(self.phase) + 1], obs)
        elif self.ticks > 1000:
            if carrying:
                self.recover(obs, "Blocked while carrying")
            else:
                raise RuntimeError(f"Blocked at piece {self.piece}, phase {self.phase}")
        action[:7] = np.clip(action[:7], self.limits[:, 0], self.limits[:, 1])
        self.last_action = action.copy()
        return action
