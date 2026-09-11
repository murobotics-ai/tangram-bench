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

from benchmark import DEFAULT_STEPS
from env import HOME, make_model
from shapes import canonical, solution
from tangram import (
    FLAT_DEGREES,
    NAMES,
    TABLE_TOLERANCE,
    THICKNESS,
    VERTICES,
    knob_yaw,
    rotation,
    score,
)
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


def goal_pose(outline, target):
    """Centre and yaw that map the canonical outline onto the painted one (exact)."""
    canon = canonical(target)[0]
    edge, ref = outline[1] - outline[0], canon[1] - canon[0]
    yaw = np.arctan2(edge[1], edge[0]) - np.arctan2(ref[1], ref[0])
    center = np.asarray(outline[0]) - rotation(yaw) @ canon[0]
    return {"goal_center": center, "goal_yaw": float(yaw)}


def via_needed(source, target):
    """A straight carry that would pass close to the base goes through VIA instead."""
    a, b = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    t = np.clip(-a @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1)
    return np.linalg.norm(a + t * (b - a)) < NEAR_BASE


def glide_ticks(start, end):
    """Joint-space glide length: the largest joint moves at GLIDE_RATE, 1.2 s at least."""
    return int(
        np.clip(np.max(np.abs(np.asarray(end) - np.asarray(start))) / GLIDE_RATE * 50, 60, GLIDE)
    )


def placement_error(pose, target):
    """Planar distance in metres and yaw difference in radians between two piece poses."""
    yaw = np.arctan2(matrix(pose[3:])[1, 0], matrix(pose[3:])[0, 0])
    goal_yaw = np.arctan2(matrix(target[3:])[1, 0], matrix(target[3:])[0, 0])
    angle = (yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(pose[:2] - target[:2])), float(abs(angle))


HEIGHTS = (0.22, 0.14)  # Candidate TCP carry heights; the lower one serves poses near the base.
HOVER = 0.03  # Piece height above the table where the final descent starts.
CARRY_SPEED = (
    0.0012  # Setpoint travel per tick while a piece is held (0.06 m/s); 0.07 and 0.08 lose grasps.
)
# Faster carries (0.12 m/s) swing the slab in the pinch and drop 21/40 to 6/40 assemblies.
FREE_SPEED = 0.003  # Setpoint travel per tick with open fingers (0.15 m/s).
ABORT_HEIGHT = 0.012  # Piece centre height at which a failed carry lets go (1 cm drop).
CLOSE_WAIT = 60  # Ticks the fingers settle on the knob before lifting (1.2 s).
RELEASE_WAIT = 50  # Ticks the hand holds still after opening (1 s).
GLIDE = 200  # Longest joint-space glide into the grasp configuration, in ticks.
GLIDE_RATE = 0.6  # rad/s of the largest joint move during the glide; sets its duration.
LIFT_SPEED = 0.0012  # Setpoint travel per tick while lifting; 0.1 m/s lost five grasps in 40.
VIA = np.array([0.5, 0.0])  # Carry via point in front of the base; used only when a leg
NEAR_BASE = 0.34  # of the direct carry would pass closer to the base than this (m).
STALL = 150  # Ticks (3 s) without progress toward the setpoint before re-planning.
STALL_CARRYING = 300  # A held piece gets 6 s: hover and lower converge slowly on purpose.
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
        self.banned = set()  # (height, turn) grasps that wound the arm up on this piece.
        self.progress = (np.inf, 0)  # (best residual in this phase, tick it was reached).
        self.repairs = 0  # Pieces re-placed after the first pass over all seven.
        self.turn = 0
        self.approach_from = None
        self.glide = GLIDE
        self.abort_position = None
        self.abandon = False  # Retries spent on the current piece: leave it for the repair pass.
        self.near = 0  # Consecutive ticks within 2 cm of the setpoint.
        self.elapsed = 0  # Control ticks since the episode began.
        self.horizon = DEFAULT_STEPS  # The protocol's horizon; a repair must fit before it.

    def choose_grasp(self, obs):
        pose = obs["pieces"][self.piece]
        yaw = np.arctan2(matrix(pose[3:])[1, 0], matrix(pose[3:])[0, 0])
        yaw += knob_yaw(VERTICES[self.piece])
        options = []
        best = np.inf  # Branch and bound: costs only grow along a path.
        original = self.data.qpos.copy()
        # The square knob admits four equivalent grasps. Check source, transit
        # and destination IK at each carry height so a convenient pickup does not
        # exhaust wrist travel or push the elbow against its limit near the base.
        candidates = [(h, t) for h in HEIGHTS for t in range(4) if (h, t) not in self.banned]
        for high, turn in candidates or [(h, t) for h in HEIGHTS for t in range(4)]:
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
                    if self.control.ik_error.max() < 1e-5:
                        break  # Converged; more Newton passes only cost time.
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
                waypoints = [
                    (np.r_[pose[:2], 0.025], orient),
                    (np.r_[pose[:2], high], orient),
                ]
                if via_needed(pose[:2], target[:2]):
                    waypoints += [(np.r_[VIA, high], orient), (np.r_[VIA, high], end_R)]
                waypoints += [(np.r_[target[:2], high], end_R), (np.r_[target[:2], 0.03], end_R)]
                # Check the path, not only its corners: the wrist can wind up between them.
                path = [(waypoints[0], True)]
                for (a, _), (b, rot) in zip(waypoints, waypoints[1:]):
                    path += [(((a + b) / 2, rot), False), ((b, rot), True)]
                for (xyz, rot), corner in path:
                    if cost > best:
                        break  # Cannot beat the best complete option: skip the rest.
                    self.control.position, self.control.orientation = xyz, rot
                    for _ in range(20):
                        cmd = self.control.action()
                        self.data.qpos[:7] = cmd[:7]
                        if self.control.ik_error.max() < 1e-5:
                            break
                    err = self.control.ik_error
                    margin = np.minimum(cmd[:7] - self.limits[:, 0], self.limits[:, 1] - cmd[:7])
                    cost += 1000 * max(0, err[0] - 0.001) + 100 * max(0, err[1] - 0.01)
                    cost += 0.1 * np.sum(1 / np.maximum(margin, 0.01))
                    # Keep well clear of the stops: a joint parked there cannot lift or turn.
                    cost += 2.0 * np.sum(np.maximum(0, 0.25 - margin) / 0.25)
                    # Close to the base the folded forearm can press on the shoulder
                    # column or the table; such a configuration is never reachable.
                    cost += 20.0 * self.clashes()
                    if corner:
                        postures.append(cmd[:7].copy())
                if cost <= best:
                    best = cost
                    options.append((cost, high, turn, orient.copy(), approach, postures))
        _, self.high, self.turn, self.grasp_orientation, self.approach, self.postures = min(
            options, key=lambda x: x[0]
        )
        self.data.qpos[:] = original
        self.source = pose[:2].copy()

    def clashes(self):
        """Robot contacts with itself or the table in the scratch state; IK cannot see them.

        Every body that is neither a piece nor the world belongs to the robot, hand
        and fingers included. Pieces are excluded because the scratch data keeps
        them at their model defaults, not where they lie on the table.
        """
        mujoco.mj_forward(self.model, self.data)
        count = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = {
                self.model.body(self.model.geom_bodyid[contact.geom1]).name,
                self.model.body(self.model.geom_bodyid[contact.geom2]).name,
            }
            robot = {b for b in bodies if b != "world" and not b.startswith("piece")}
            if bodies == {"left_finger", "right_finger"}:
                continue  # Closed fingers touch each other by design.
            if robot and bodies <= robot | {"world"}:
                count += 1
        return count

    def misplaced(self, obs):
        """The piece to re-place, or None: the worst one that is off its certificate
        pose, standing high (on a neighbour) or tilted, i.e. anything the scorer's
        physical gates or placement tolerance would reject."""
        worst, largest = None, 0.0
        for i in range(7):
            pose = obs["pieces"][i]
            distance, angle = placement_error(pose, self.targets[i])
            height = abs(pose[2] - THICKNESS / 2)
            tilt = np.arccos(np.clip(1 - 2 * (pose[4] ** 2 + pose[5] ** 2), -1, 1))
            off = (
                distance > 0.0025
                or angle > np.deg2rad(2.5)
                or height >= TABLE_TOLERANCE
                or tilt >= np.deg2rad(FLAT_DEGREES)
            )
            badness = distance + 0.1 * angle + 10 * height + 0.5 * tilt
            if off and badness > largest:
                worst, largest = i, badness
        return worst

    def finish_retreat(self, obs):
        """The hand is clear of the piece: advance, or plan the same piece again.

        "Settled" uses the scorer's own table tolerance, and an assembly the scorer
        accepts always advances: a released slab a few millimetres up on a neighbour
        is a success by the benchmark's test and must not be picked up again.
        """
        settled = obs["pieces"][self.piece][2] < THICKNESS / 2 + TABLE_TOLERANCE
        accepted = score(obs["pieces"], obs["piece_velocities"], obs["goal"])["success"]
        if self.regrasp:
            pass  # Pick the same piece up again from where it lies.
        elif settled or accepted or self.abandon:
            self.index += 1
            self.retries = 0
            self.banned.clear()
        else:
            # Landed on a neighbour's edge: pick it up and place it again, or leave it
            # to the repair pass once three attempts are spent.
            self.retries += 1
            self.retries_total += 1
            if self.retries > 3:
                self.index += 1
                self.retries = 0
                self.banned.clear()
        self.regrasp, self.descents, self.abandon = False, 0, False
        self.transition("start", obs)

    def describe(self, obs):
        """One sentence saying what the controller is doing now, for language annotations."""
        if self.phase == "done":
            return "All seven pieces are placed; hold still."
        name, figure = NAMES[self.piece], obs["target"]
        where = region(self.targets[self.piece][:2], obs["goal"], self.yaw)
        if self.phase == "abort":
            return f"The {name} is not held well; lower it to the table and let go."
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
        self.phase, self.ticks, self.stable, self.near = phase, 0, 0, 0
        self.setpoint = obs["tcp_pos"].copy()
        self.rate = 0.0
        self.progress = (np.inf, 0)

    def recover(self, obs, reason):
        """Let go, back off and pick the same piece again from where it lies now."""
        self.retries += 1
        self.retries_total += 1
        # After three attempts, let go and move on: the repair pass at the end comes
        # back to whatever is still off, often after its neighbours have been fixed.
        # Ending the episode here would give up on the time that is left.
        self.abandon = self.retries > 3
        self.regrasp = not self.abandon
        self.banned.add((self.high, self.turn))  # Try another of the knob's four grasps.
        piece_z = obs["pieces"][self.piece][2]
        if self.phase in ("lift", "transit", "transfer", "hover", "lower") and piece_z > 0.03:
            # Dropping a slab from carry height flips it onto its edge or its back,
            # and a knob facing down cannot be grasped again: bring it down first.
            drop = piece_z - ABORT_HEIGHT
            self.abort_position = obs["tcp_pos"] - [0, 0, drop]
            self.transition("abort", obs)
            return
        self.release_joints = self.last_action[:7].copy()
        self.release_position, self.release_R = obs["tcp_pos"].copy(), obs["tcp_mat"].copy()
        self.transition("release", obs)

    def repair_fits(self):
        """A repair takes about 40 s; one that cannot finish leaves a piece in the air."""
        return self.horizon - self.elapsed > 2000

    def act(self, obs):
        self.elapsed += 1
        if self.targets is None:
            # Read the goal pose off the painted outline itself, so an overridden
            # scene (tools.collect --goal-yaw) gets the right certificate too.
            scene = goal_pose(obs["goal"], obs["target"])
            self.targets = solution(obs["target"], self.seed, scene)
            self.yaw = scene["goal_yaw"]
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
            if self.index >= 7:
                # First pass complete. A piece disturbed by a later placement is the
                # most common failure: re-place the worst one instead of holding
                # still for the rest of the horizon. An assembly the benchmark's own
                # test already accepts is never touched again, whatever the
                # certificate says: the score is the goal, not the certificate.
                self.index = 7
                accepted = score(obs["pieces"], obs["piece_velocities"], obs["goal"])["success"]
                worst = None if accepted else self.misplaced(obs)
                if worst is None or self.repairs >= 5 or not self.repair_fits():
                    self.approach_from = self.last_action[:7].copy()
                    self.glide = glide_ticks(self.approach_from, HOME["panda"])
                    self.transition("done", obs)
                else:
                    self.repairs += 1
                    self.piece, self.retries = worst, 0
                    self.banned.clear()
                    self.choose_grasp(obs)
                    self.approach_from = obs["qpos"][:7].copy()
                    self.glide = glide_ticks(self.approach_from, self.approach)
                    self.transition("approach", obs)
            else:
                self.piece = self.order[self.index]
                self.choose_grasp(obs)
                self.approach_from = obs["qpos"][:7].copy()
                self.glide = glide_ticks(self.approach_from, self.approach)
                self.transition("approach", obs)
        self.step = min(self.index + 1, len(self.order))
        self.subtask = self.describe(obs)
        if self.phase == "done":
            # Glide back to the home configuration and hold it there: the episode
            # ends with the arm out of the way, as a demonstration should.
            # Keep watching: a slab resting on a neighbour's edge can creep for tens
            # of seconds and slide out of tolerance; repair while repairs remain.
            self.ticks += 1
            if self.ticks % 50 == 0 and self.repairs < 5 and self.repair_fits():
                if not score(obs["pieces"], obs["piece_velocities"], obs["goal"])["success"]:
                    self.index = 7
                    self.transition("start", obs)
                    return self.act(obs)
            home = np.array(HOME["panda"])
            s = min(1.0, self.ticks / self.glide)
            s = s * s * (3 - 2 * s)
            action = np.r_[self.approach_from + s * (home - self.approach_from), 1.0]
            self.last_action = action.copy()
            return action

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
        # A carried piece never needs the hand far above the carry height or tilted
        # off vertical: either means the wrist wound up against its stops. Let go now
        # and pick the piece again with another grasp instead of hanging for seconds.
        if self.phase in ("lift", "transit", "transfer", "hover", "lower") and (
            obs["tcp_pos"][2] > self.high + 0.08 or obs["tcp_mat"][2, 2] > -0.5
        ):
            self.recover(obs, "Arm wound up while carrying")
        if self.phase == "release":
            self.ticks += 1
            action = np.r_[self.release_joints, 1.0]
            if self.ticks >= RELEASE_WAIT:
                self.release_position = obs["tcp_pos"].copy()
                self.release_R = obs["tcp_mat"].copy()
                self.transition("retreat", obs)
            self.last_action = action.copy()
            return action
        via = len(self.postures) == 6  # choose_grasp added the via-point waypoints
        posture_index = {
            "descend": 0,
            "close": 0,
            "lift": 1,
            "transit": 3 if via else 2,
            "transfer": 4 if via else 2,
            "hover": 5 if via else 3,
            "lower": 5 if via else 3,
            "abort": 5 if via else 3,
            "retreat": 4 if via else 2,
        }.get(self.phase)
        # Steer the redundant joint toward the configuration checked for this phase,
        # so the arm follows the collision-free branch choose_grasp found.
        self.control.posture = np.array(HOME["panda"])
        if posture_index is not None:
            self.control.posture = self.postures[posture_index]
        pose = obs["pieces"][self.piece]
        target = self.targets[self.piece]
        R = self.grasp_orientation
        carrying = self.phase in ("lift", "transit", "transfer", "hover", "lower", "abort")
        grip = 0.0 if self.phase == "close" or carrying else 1.0
        if self.phase in ("approach", "descend", "close", "lift"):
            position = np.r_[
                self.source, self.high if self.phase in ("approach", "lift") else 0.025
            ]
            if self.phase == "lift" and np.linalg.norm(self.source) < 0.32:
                # Close to the base the elbow is folded to its stop: lift outward
                # along the diagonal, where the arm has room, not straight up.
                position[:2] += 0.06 * self.source / np.linalg.norm(self.source)
        elif self.phase in ("transit", "transfer", "hover", "lower"):
            # Measure the grasp again each tick to correct small slips, but in yaw only.
            # The pinch on the knob lets the slab pivot about the finger axis; tilting
            # the hand to "correct" a measured tilt chases that pivot and winds the
            # wrist up against its stops. The hand stays vertical; the slab flattens
            # when it meets the table.
            piece_R, target_R = matrix(pose[3:]), matrix(target[3:])
            hand_yaw = np.arctan2(obs["tcp_mat"][1, 0], obs["tcp_mat"][0, 0])
            piece_yaw = np.arctan2(piece_R[1, 0], piece_R[0, 0])
            target_yaw = np.arctan2(target_R[1, 0], target_R[0, 0])
            relative_p = obs["tcp_mat"].T @ (pose[:3] - obs["tcp_pos"])
            R = downward(hand_yaw + target_yaw - piece_yaw)
            desired_piece = target[:3].copy()
            if self.phase == "transit" and via_needed(self.source, target[:2]):
                desired_piece[:2] = VIA  # Swing wide of the base; otherwise go straight.
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
        elif self.phase == "abort":
            # Straight down with the hand vertical, then release near the table.
            position = self.abort_position.copy()
            R = downward(np.arctan2(obs["tcp_mat"][1, 0], obs["tcp_mat"][0, 0]))
        else:
            position, R = self.release_position.copy(), self.release_R
            if self.phase == "retreat":
                # Back off to the carry height whose reachability was checked.
                position[2] = max(position[2], self.high)
        residual = position - obs["tcp_pos"]
        speed = FREE_SPEED
        if self.phase in ("transit", "transfer", "hover"):
            speed = CARRY_SPEED
        elif self.phase == "lift":
            speed = LIFT_SPEED
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
        if self.phase == "approach" and self.ticks < self.glide:
            # Glide in joint space to the chosen configuration; a step command jerks the arm.
            s = (self.ticks + 1) / self.glide
            s = s * s * (3 - 2 * s)
            action = np.r_[self.approach_from + s * (self.approach - self.approach_from), 1.0]
            self.setpoint = obs["tcp_pos"].copy()
        self.ticks += 1
        tolerance = {"lower": (0.0015, 0.02), "hover": (0.002, 0.03)}.get(self.phase, (0.006, 0.06))
        near = np.linalg.norm(residual) < tolerance[0]
        aligned = np.linalg.norm(error) < tolerance[1]
        self.near = self.near + 1 if np.linalg.norm(residual) < 0.02 else 0
        self.stable = self.stable + 1 if near and aligned else 0
        wait = CLOSE_WAIT if self.phase == "close" else 8
        distance = np.linalg.norm(residual) + 0.05 * np.linalg.norm(error)
        if distance < self.progress[0] - 0.001:
            self.progress = (distance, self.ticks)
        if self.phase == "abort" and (
            self.stable >= wait or self.ticks - self.progress[1] > STALL or self.ticks > 300
        ):
            self.release_joints = action[:7].copy()
            self.release_position, self.release_R = obs["tcp_pos"].copy(), obs["tcp_mat"].copy()
            self.transition("release", obs)
        elif self.stable >= wait:
            if self.phase == "lower":
                self.release_joints = action[:7].copy()
            if self.phase == "lift" and pose[2] < 0.08:
                self.recover(obs, "Grasp failed")
            elif self.phase == "retreat":
                self.finish_retreat(obs)
            else:
                self.transition(PHASES[PHASES.index(self.phase) + 1], obs)
        else:
            # Stall detection: a phase that has not brought the hand a millimetre
            # closer in STALL ticks is stuck (self-collision, a stop, a wedged
            # slab). Let go and plan the piece again now, instead of standing still
            # for the hard timeout; a hand parked for 20 s is useless as a demo.
            # Progress (updated above) counts the orientation too, 0.05 m per radian,
            # so a hand that is in place but still turning is not mistaken for a stall.
            limit = STALL_CARRYING if carrying else STALL
            gliding = self.phase == "approach" and self.ticks < self.glide + STALL
            stalled = (
                self.ticks - self.progress[1] > limit and self.phase != "close" and not gliding
            )
            if stalled and self.phase == "retreat":
                # The retreat cannot reach its height from here (a wound wrist after
                # the descent); the next grasp's joint-space glide gets the arm out.
                self.finish_retreat(obs)
            elif self.phase in ("transit", "transfer") and self.near >= STALL:
                # A carry waypoint reached to within 2 cm is reached: its precision does
                # not matter, hover corrects it. Waiting for the last millimetre against
                # a joint stop cost 20 s per occurrence. Approach and lift keep their
                # tolerance: a grasp 2 cm off or a half-lifted slab both slip.
                self.transition(PHASES[PHASES.index(self.phase) + 1], obs)
            elif stalled or self.ticks > 1000:
                self.recover(obs, f"{'Stalled' if stalled else 'Blocked'} in {self.phase}")
        action[:7] = np.clip(action[:7], self.limits[:, 0], self.limits[:, 1])
        self.last_action = action.copy()
        return action
