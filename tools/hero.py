"""Render the README picture: ten arms collecting demonstrations, the collector's fleet view.

Usage: MUJOCO_GL=egl uv run -m tools.hero            # writes assets/preview.png
       MUJOCO_GL=egl uv run -m tools.hero out.png

Ten seeds over the three training figures are simulated with the reference
controller to different points of their episodes, one process each, and drawn
in one hall with `env.fleet_model`, the way `tools.collect --workers 10 --watch
fleet` shows a live round, with each arm's figure, seed, plan step and subtask
in the overlay. Needs an offscreen GL backend (MUJOCO_GL=egl or osmesa).
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from env import Env, draw_goal, fleet_model
from eval import load_policy

ARMS = [  # (seed, figure, control ticks into the episode)
    (3, "house", 1800),
    (11, "square", 4200),
    (7, "rectangle", 6500),
    (21, "house", 3000),
    (5, "square", 7800),
    (14, "rectangle", 2400),
    (9, "house", 5600),
    (2, "square", 1200),
    (17, "rectangle", 4800),
    (8, "house", 7000),
]
COLUMNS = 5
SIZE = (1680, 900)
CAMERA = {"distance": 8.4, "azimuth": 90, "elevation": -33, "lookat": (0.3, 0.5)}


def simulate(arm):
    """State of one arm `ticks` control steps into its episode, with its outline and label."""
    seed, target, ticks = arm
    env = Env("panda", pixels=False)
    policy = load_policy("examples/oracle.py")(robot="panda", seed=seed)
    obs = env.reset([seed], [target])[0]
    for _ in range(ticks):
        action = np.asarray(policy.act({**obs, "images": {}}), dtype=float).reshape(1, -1)
        obs = env.step([env.validate_actions(action)[0]])[0]
    state = {
        "seed": seed,
        "target": target,
        "qpos": env.data[0].qpos.copy(),
        "goal": np.asarray(obs["goal"]),
        "label": str(policy.subtask).removesuffix(" of the outline."),
        "step": int(policy.step),
        "plan": len(policy.plan),
    }
    env.close()
    return state


def render(arms, path):
    from PIL import Image

    model, offsets, index, shift = fleet_model("panda", len(arms), COLUMNS)
    data = mujoco.MjData(model)
    for i, arm in enumerate(arms):
        data.qpos[index[i]] = arm["qpos"] + shift @ offsets[i]
    mujoco.mj_forward(model, data)
    width, height = SIZE
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    gl = mujoco.GLContext(width, height)
    gl.make_current()
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    try:
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        scene = mujoco.MjvScene(model, maxgeom=4000)
        camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        center = offsets.mean(axis=0)
        camera.lookat[:] = [center[0] + CAMERA["lookat"][0], center[1] + CAMERA["lookat"][1], 0.1]
        camera.distance, camera.azimuth, camera.elevation = (
            CAMERA["distance"],
            CAMERA["azimuth"],
            CAMERA["elevation"],
        )
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        for arm, offset in zip(arms, offsets):
            draw_goal(scene, arm["goal"], offset)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, scene, context)
        lines = [
            f"ARM {i + 1:2d}  {a['target']:<9s} seed {a['seed']:<3d} {a['step']}/{a['plan']}  {a['label']}"
            for i, a in enumerate(arms)
        ]
        half = (len(lines) + 1) // 2
        for chunk, corner in (
            (lines[:half], mujoco.mjtGridPos.mjGRID_TOPLEFT),
            (lines[half:], mujoco.mjtGridPos.mjGRID_TOPRIGHT),
        ):
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL, corner, viewport, "\n".join(chunk), "", context
            )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            "tools.collect --workers 10 --watch fleet | ten arms, ten seeds, one dataset",
            "",
            context,
        )
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, viewport, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.flipud(pixels)).save(path)
    finally:
        context.free()
        gl.free()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else Path("assets") / "preview.png"
    start = time.time()
    with Pool(len(ARMS)) as pool:
        arms = pool.map(simulate, ARMS)
    render(arms, path)
    print(f"{path}: {len(arms)} arms, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
