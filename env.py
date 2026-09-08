"""A small physical environment. Policies receive copies, never simulator handles.

Both backends consume the same MJCF and bounded position-actuator targets.
Warp batches physics on GPU; observation transfer and scoring remain on CPU in v0.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tangram import (
    CENTERS,
    COLORS,
    KNOB_HEIGHT,
    KNOB_WIDTH,
    SIDE,
    THICKNESS,
    VERTICES,
    goal,
    knob_yaw,
    rotation,
)
from tools.prepare import ASSETS, ROBOTS

DT = 0.002
SUBSTEPS = 10  # 50 Hz policy, 500 Hz physics.
HOME = {"panda": [0, -0.45, 0, -2.2, 0, 1.8, 0.7854], "piper": [0, 1.0, -1.0, 0, 0.5, 0]}


def camera_axes(position, target):
    """MuJoCo cameras look along local -Z, with local +Y up."""
    z = np.asarray(position, dtype=float) - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    return " ".join(map(str, np.r_[x, np.cross(z, x)]))


def make_model(robot="panda"):
    source = ASSETS / ROBOTS[robot]
    if not source.exists():
        raise FileNotFoundError("Robot assets missing. Run: uv run -m tools.prepare")
    root = ET.parse(source).getroot()
    # Resolve mesh paths before compiling from a string; preserve upstream licenses.
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)  # Robot-only keyframes have the wrong nq after adding pieces.
    # Menagerie grippers are soft position servos: their squeeze on a 2 cm knob is the
    # gap times the stiffness (1 N Panda, 0.4 N PiPER), too little to lift a piece.
    # Stiffen them tenfold; the command mapping (ctrlrange) is unchanged.
    gripper = root.find(".//actuator/*[@name='actuator8']")
    if gripper is not None:
        gripper.set("gainprm", "0.1568627451 0 0")
        gripper.set("biasprm", "0 -1000 -100")
    gripper = root.find(".//actuator/*[@name='gripper']")
    if gripper is not None:
        gripper.set("kp", "400")
        gripper.set("kv", "20")
        gripper.set("forcerange", "-40 40")
    option = root.find("option")
    option.set("timestep", str(DT))
    option.set("solver", "Newton")
    option.set("iterations", "50")
    option.set("cone", "elliptic")
    world = root.find("worldbody")
    # Lighting is part of the benchmark scene, so drop the robot file's own light and
    # define every source here. These values shape future pixel observations.
    for light in world.findall("light"):
        world.remove(light)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual, "headlight", ambient=".15 .15 .15", diffuse=".35 .35 .35", specular=".1 .1 .1"
    )
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        size="1 1 .05",
        pos="0 0 -.05",
        rgba=".89 .88 .85 1",
        friction=".7 .005 .0001",
    )
    ET.SubElement(
        world,
        "light",
        name="key",
        pos=".3 -.4 1.5",
        dir="0 .2 -1",
        diffuse=".7 .7 .7",
        specular=".1 .1 .1",
        ambient=".05 .05 .05",
    )
    # Frame the whole table in a 4:3 top view: 1.3 m up, fovy 44 covers x in [-0.40, 1.00], y in [-0.53, 0.53].
    ET.SubElement(world, "camera", name="top", pos=".3 0 1.3", quat="1 0 0 0", fovy="44")
    ET.SubElement(
        world,
        "camera",
        name="context",
        pos="1.1 -1.35 1.05",
        fovy="50",
        xyaxes=camera_axes([1.1, -1.35, 1.05], [0.28, -0.10, 0.23]),
    )
    parent = world.find(".//body[@name='hand']" if robot == "panda" else ".//body[@name='link6']")
    ET.SubElement(
        parent,
        "site",
        name="tcp",
        pos="0 0 .1034" if robot == "panda" else "0 0 .18",
        size=".003",
        rgba="0 0 0 0",
    )
    # Lens on the top face of the gripper body, behind the fingers, looking along the
    # tool axis so the fingertips stay in the lower part of the frame. The PiPER lens
    # sits on its -x face (upward at home) with a narrower view that keeps the body out.
    wrist_pos = [0.065, 0, 0.04] if robot == "panda" else [-0.09, 0, 0.06]
    wrist_aim = [0, 0, 0.25] if robot == "panda" else [0, 0, 0.30]
    ET.SubElement(
        parent,
        "camera",
        name="wrist",
        pos=" ".join(map(str, wrist_pos)),
        fovy="75" if robot == "panda" else "60",
        xyaxes=camera_axes(wrist_pos, wrist_aim),
    )
    asset = root.find("asset")
    for i, vertices in enumerate(VERTICES):
        prism = [[x, y, z] for z in [-THICKNESS / 2, THICKNESS / 2] for x, y in vertices]
        ET.SubElement(asset, "mesh", name=f"tile{i}", vertex=" ".join(map(str, np.ravel(prism))))
        body = ET.SubElement(world, "body", name=f"piece{i}", pos=f".4 {-0.3 + i * 0.1} .1")
        ET.SubElement(body, "freejoint", name=f"piece{i}")
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=f"tile{i}",
            rgba=COLORS[i],
            density="700",
            friction=".8 .01 .0001",
            condim="4",
        )
        yaw = knob_yaw(vertices)
        ET.SubElement(
            body,
            "geom",
            name=f"knob{i}",
            type="box",
            size=f"{KNOB_WIDTH / 2} {KNOB_WIDTH / 2} {KNOB_HEIGHT / 2}",
            pos=f"0 0 {THICKNESS / 2 + KNOB_HEIGHT / 2}",
            quat=f"{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}",
            rgba=COLORS[i],
            density="700",
            friction=".8 .01 .0001",
            condim="4",
        )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class Env:
    def __init__(self, robot="panda", backend="cpu", num_envs=1):
        if robot not in ROBOTS or backend not in ("cpu", "warp") or num_envs < 1:
            raise ValueError("Invalid robot, backend, or num_envs")
        self.robot, self.backend, self.num_envs = robot, backend, num_envs
        self.model = make_model(robot)
        self.narm = len(HOME[robot])
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.qadr = [self.model.joint(f"piece{i}").qposadr[0] for i in range(7)]
        self.vadr = [self.model.joint(f"piece{i}").dofadr[0] for i in range(7)]
        self.tcp = self.model.site("tcp").id
        self.limits = self.model.actuator_ctrlrange.copy()
        self.steps = 0
        self.graph = None
        if backend == "warp":
            import mujoco_warp as mjw
            import warp as wp

            wp.config.quiet = True  # Keep kernel-cache chatter out of experiment logs.
            if not wp.is_cuda_available():
                raise RuntimeError(
                    "MJWarp evaluation requires an NVIDIA CUDA GPU; use --backend cpu."
                )
            print(
                f"MJWarp: {wp.get_device().name}; preparing kernels (first run can take minutes).",
                flush=True,
            )
            self.wp, self.mjw = wp, mjw
            self.wm = mjw.put_model(self.model)
            self.wd = mjw.make_data(self.model, nworld=num_envs, nconmax=256, njmax=1024)

    def reset(self, seeds):
        if len(seeds) != self.num_envs or any(int(s) < 0 for s in seeds):
            raise ValueError("Provide one nonnegative seed per world")
        self.outlines = [goal(int(s)) for s in seeds]
        self.steps = 0
        for d, seed in zip(self.data, seeds):
            mujoco.mj_resetData(self.model, d)
            d.qpos[: self.narm] = HOME[self.robot]
            d.qpos[self.narm : self.narm + 2] = (
                [0.04, 0.04] if self.robot == "panda" else [0.035, -0.035]
            )
            d.ctrl[: self.narm] = HOME[self.robot]
            d.ctrl[-1] = self.limits[-1, 1]
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
            center = rng.uniform([0.28, -0.30], [0.36, -0.24])
            yaw = rng.uniform(-np.pi, np.pi)
            xy = ((CENTERS - 0.5) * SIDE) @ rotation(yaw).T + center
            # Move the complete dissection rigidly; independent piece yaw breaks the packing.
            # Start on the table: dropping touching tiles can disturb their shared edges.
            for address, position in zip(self.qadr, xy):
                d.qpos[address : address + 7] = [
                    *position,
                    THICKNESS / 2,
                    np.cos(yaw / 2),
                    0,
                    0,
                    np.sin(yaw / 2),
                ]
            mujoco.mj_forward(self.model, d)
            # The same CPU settling procedure initializes both backends.
            mujoco.mj_step(self.model, d, nstep=100)
        self.previous = np.stack([d.ctrl.copy() for d in self.data])
        if self.backend == "warp":
            for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                getattr(self.wd, name).assign(np.stack([getattr(d, name) for d in self.data]))
            self.wd.time.assign(np.array([d.time for d in self.data]))
            self.mjw.forward(self.wm, self.wd)
            if self.graph is None:
                # Compile once outside capture. Reset the state after warmup.
                self.mjw.step(self.wm, self.wd)
                self.wp.synchronize()
                for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                    getattr(self.wd, name).assign(np.stack([getattr(d, name) for d in self.data]))
                self.wd.time.assign(np.array([d.time for d in self.data]))
                self.mjw.forward(self.wm, self.wd)
                with self.wp.ScopedCapture() as capture:
                    for _ in range(SUBSTEPS):
                        self.mjw.step(self.wm, self.wd)
                self.graph = capture.graph
        return self.observe()

    def observe(self):
        if self.backend == "warp":
            # Only transfer fields needed by the state-only protocol.
            qp, qv = self.wd.qpos.numpy(), self.wd.qvel.numpy()
            for i, d in enumerate(self.data):
                d.qpos[:], d.qvel[:] = qp[i], qv[i]
        observations = []
        for d, outline in zip(self.data, self.outlines):
            # Refresh tool/geometry poses without rerunning CPU contact dynamics.
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            observations.append(
                {
                    "robot": self.robot,
                    "time": self.steps * DT * SUBSTEPS,
                    "qpos": d.qpos[: self.narm + 2].copy(),
                    "qvel": d.qvel[: self.narm + 2].copy(),
                    "tcp_pos": d.site_xpos[self.tcp].copy(),
                    "tcp_mat": d.site_xmat[self.tcp].reshape(3, 3).copy(),
                    "pieces": np.stack([d.qpos[a : a + 7] for a in self.qadr]),
                    "piece_velocities": np.stack([d.qvel[a : a + 6] for a in self.vadr]),
                    "goal": outline.copy(),
                }
            )
        return observations

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_envs, self.narm + 1) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions of shape {(self.num_envs, self.narm + 1)}")
        if np.any(actions[:, :-1] < self.limits[:-1, 0]) or np.any(
            actions[:, :-1] > self.limits[:-1, 1]
        ):
            raise ValueError("Arm action exceeds actuator limits")
        if np.any((actions[:, -1] < 0) | (actions[:, -1] > 1)):
            raise ValueError("Gripper opening must be in [0, 1]")
        controls = actions.copy()
        controls[:, -1] = self.limits[-1, 0] + actions[:, -1] * np.diff(self.limits[-1])[0]
        # Fixed 2 rad/s target slew limit, shared by all policies and backends.
        controls[:, :-1] = np.clip(
            controls[:, :-1], self.previous[:, :-1] - 0.04, self.previous[:, :-1] + 0.04
        )
        self.previous = controls.copy()
        if self.backend == "cpu":
            for d, ctrl in zip(self.data, controls):
                d.ctrl[:] = ctrl
                mujoco.mj_step(self.model, d, nstep=SUBSTEPS)
        else:
            self.wd.ctrl.assign(controls)
            self.wp.capture_launch(self.graph)
        for d in self.data:
            if any(
                d.warning[w].number
                for w in (
                    mujoco.mjtWarning.mjWARN_BADQPOS,
                    mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC,
                )
            ):
                raise RuntimeError("MuJoCo reported unstable dynamics; episode cannot be scored")
        self.steps += 1
        obs = self.observe()
        if not all(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() for d in self.data):
            raise RuntimeError("Non-finite simulation state")
        return obs
