"""Scene inspection and saved episode replay with goal, top and wrist panels.

Static by default; --teleop enables manual physics. Never executes a policy.
"""

import argparse
import textwrap
import time
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import shapely

from env import Env
from shapes import PROMPT, TARGETS
from tangram import goal

CAMERAS = ("context", "wrist", "top")
PROMPT_HEIGHT = 96


def target_outline(name, seed):
    if name not in TARGETS:
        raise ValueError(f"Target {name!r} is not defined; available: {', '.join(TARGETS)}")
    return goal(seed, name)


def panel_text(selected, camera):
    return (
        f"TANGRAM-BENCH  |  {selected.title()}\n"
        f"Main: {camera}  |  1 context / 2 wrist / 3 top\n"
        "Policy input: exact state + optional prompt\n"
        "Cameras: inspection views"
    )


def layout(width, height):
    """Disjoint framebuffer rectangles, with OpenGL's bottom-left origin."""
    height -= PROMPT_HEIGHT
    sidebar = width // 3
    main = mujoco.MjrRect(0, 0, width - sidebar, height)
    x, w = main.width + 8, sidebar - 16
    h = (height - 32) // 3
    panels = [mujoco.MjrRect(x, height - 8 - (i + 1) * h - i * 8, w, h) for i in range(3)]
    return main, panels


def image_rect(panel):
    """One 4:3 image viewport, identical for goal, top and wrist, under a 28 px title."""
    w, h = panel.width, panel.height - 28
    if w * 3 > h * 4:
        w = h * 4 // 3
    else:
        h = w * 3 // 4
    return mujoco.MjrRect(
        panel.left + (panel.width - w) // 2, panel.bottom + (panel.height - 28 - h) // 2, w, h
    )


@lru_cache(maxsize=4)
def goal_image(vertices, width, height):
    """Fit the silhouette without changing its orientation or aspect ratio."""
    outline = np.array(vertices)
    center = (outline.max(axis=0) + outline.min(axis=0)) / 2
    scale = min(width, height) * 0.72 / np.ptp(outline, axis=0).max()
    polygon = shapely.Polygon((outline - center) * scale + [width / 2, height / 2])
    y, x = np.mgrid[:height, :width]
    pixels = np.full((height, width, 3), 246, dtype=np.uint8)
    pixels[shapely.contains_xy(polygon, x + 0.5, y + 0.5)] = [25, 29, 34]
    return pixels


def draw_goal(scene, outline):
    """Exact concave silhouette triangles, visual only; never add collision geometry."""
    for triangle in goal_triangles(tuple(map(tuple, outline))):
        a, b, c = triangle
        # The renderer's triangle is (0,0), (1,0), (0,1); this affine basis maps it exactly.
        mat = np.column_stack((np.r_[b - a, 0], np.r_[c - a, 0], [0, 0, 1]))
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_TRIANGLE,
            np.ones(3),
            np.r_[a, 0.0002],
            mat.ravel(),
            np.array([0.12, 0.14, 0.16, 1]),
        )
        scene.ngeom += 1


@lru_cache(maxsize=64)
def goal_triangles(vertices):
    from shapely.ops import triangulate

    polygon = shapely.Polygon(vertices)
    triangles = [p for p in triangulate(polygon) if polygon.covers(p)]
    if abs(sum(p.area for p in triangles) - polygon.area) > 1e-10:
        raise ValueError("Silhouette triangulation does not cover the goal")
    return [np.array(p.exterior.coords)[:3] for p in triangles]


def set_camera(camera, model, name):
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = model.camera(name).id


def draw_frame(env, scene, context, width, height, camera, target, outline, control=None):
    """All cameras see the same state and the visual-only target on the table."""
    main, panels = layout(width, height)
    mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0.055, 0.065, 0.08, 1)
    prompt_rect = mujoco.MjrRect(8, height - PROMPT_HEIGHT + 8, width - 16, PROMPT_HEIGHT - 16)
    mujoco.mjr_rectangle(prompt_rect, 0.13, 0.15, 0.18, 1)
    prompt = textwrap.fill(env.prompt, width=max(30, (width - 40) // 12))
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        prompt_rect,
        "TASK PROMPT\n" + prompt,
        "",
        context,
    )
    viewport = image_rect(panels[0])
    pixels = goal_image(tuple(map(tuple, outline)), viewport.width, viewport.height)
    mujoco.mjr_drawPixels(pixels.ravel(), None, viewport, context)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    for name, rect in [(camera, main), ("top", panels[1]), ("wrist", panels[2])]:
        viewport = rect if rect is main else image_rect(rect)
        set_camera(cam, env.model, name)
        mujoco.mjv_updateScene(
            env.model, env.data[0], opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        draw_goal(scene, outline)
        mujoco.mjr_render(viewport, scene, context)
    titles = (
        f"GOAL  /  {target.title()}  (not seen by policy)",
        "TOP CAMERA",
        "WRIST CAMERA",
    )
    for rect, title in zip(panels, titles):
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, rect, title, "", context
        )
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        main,
        (env.text() if hasattr(env, "seek") else panel_text(target, camera))
        if control is None
        else (f"TANGRAM-BENCH | {target.title()} | 1/2/3: main camera\n" + control.text()),
        "",
        context,
    )


def render_scene(env, outline, camera, target, path):
    from PIL import Image

    width, height = 1440, 960
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = width, height
    gl = mujoco.GLContext(width, height)
    gl.make_current()
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    try:
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        scene = mujoco.MjvScene(env.model, maxgeom=1000)
        draw_frame(env, scene, context, width, height, camera, target, outline)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, width, height), context)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.flipud(pixels)).save(path)
    finally:
        context.free()
        gl.free()


def show_scene(env, outline, camera, target, teleop=False, seed=100000):
    import glfw

    from teleop import CONTROL_DT, Teleop

    control = Teleop(env) if teleop else None
    playback = hasattr(env, "seek")
    paused = False
    if not glfw.init():
        raise RuntimeError("Cannot open display; use MUJOCO_GL=egl and --out for a PNG.")
    window = None
    context = None
    try:
        window = glfw.create_window(1440, 960, f"Tangram-Bench | {env.robot}", None, None)
        if not window:
            raise RuntimeError("Could not create the viewer window")
        glfw.set_window_size_limits(window, 720, 600, glfw.DONT_CARE, glfw.DONT_CARE)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        scene = mujoco.MjvScene(env.model, maxgeom=1000)

        def key_callback(window, key, scancode, action, mods):
            nonlocal camera, control, paused
            if action != glfw.PRESS:
                return
            if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
                camera = CAMERAS[key - glfw.KEY_1]
            elif control is not None and key == glfw.KEY_P:
                control.paused = not control.paused
            elif control is not None and key == glfw.KEY_O:
                env.reset([seed], [target], env.prompt)  # SpaceMouse right button
                control = Teleop(env)
            elif playback and key == glfw.KEY_SPACE:
                paused = not paused
            elif playback and key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                env.seek(env.steps + (50 if key == glfw.KEY_RIGHT else -50))
            elif playback and key in (glfw.KEY_HOME, glfw.KEY_END):
                env.seek(0 if key == glfw.KEY_HOME else env.frames - 1)
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

        glfw.set_key_callback(window, key_callback)
        last, accumulator = time.monotonic(), 0.0
        while not glfw.window_should_close(window):
            now = time.monotonic()
            accumulator += min(now - last, 0.08)
            last = now
            if control is not None:

                def pressed(key):
                    return int(
                        glfw.get_window_attrib(window, glfw.FOCUSED)
                        and glfw.get_key(window, key) == glfw.PRESS
                    )

                def axis(positive, negative):
                    return pressed(positive) - pressed(negative)

                # Isaac Lab's Se3Keyboard axes, with the gripper and reset moved to the left hand.
                while accumulator >= CONTROL_DT:
                    control.step(
                        [
                            axis(glfw.KEY_W, glfw.KEY_S),
                            axis(glfw.KEY_A, glfw.KEY_D),
                            axis(glfw.KEY_Q, glfw.KEY_E),
                        ],
                        [
                            axis(glfw.KEY_Z, glfw.KEY_X),
                            axis(glfw.KEY_T, glfw.KEY_G),
                            axis(glfw.KEY_C, glfw.KEY_V),
                        ],
                        axis(glfw.KEY_R, glfw.KEY_F),  # Hold: R opens, F closes, progressively.
                        pressed(glfw.KEY_LEFT_SHIFT) or pressed(glfw.KEY_RIGHT_SHIFT),
                    )
                    accumulator -= CONTROL_DT
            elif playback and not paused:
                while accumulator >= CONTROL_DT:
                    env.seek(env.steps + 1)
                    accumulator -= CONTROL_DT
                if env.steps == env.frames - 1:
                    paused = True
            else:
                accumulator = 0

            width, height = glfw.get_framebuffer_size(window)
            if width >= 720 and height >= 600:
                draw_frame(env, scene, context, width, height, camera, target, outline, control)
                glfw.swap_buffers(window)
            glfw.wait_events_timeout(1 / 30)
    finally:
        if context is not None:
            context.free()
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=["panda", "piper"], default="panda")
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--backend", choices=["cpu", "warp"], default="cpu")
    p.add_argument("--target", choices=TARGETS, default="square")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument(
        "--camera",
        choices=CAMERAS,
        default="context",
        help="Main view; top/wrist panels remain visible",
    )
    p.add_argument(
        "--teleop", action="store_true", help="Manual TCP and gripper control; no recording"
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--result", type=Path, help="Replay a saved evaluation")
    p.add_argument("--episode", type=int, default=0, help="Zero-based episode index in --result")
    p.add_argument("--frame", type=int, default=0, help="Initial replay frame")
    args = p.parse_args()
    if args.teleop and args.out:
        p.error("--teleop is interactive; use --out without --teleop for a static PNG")
    if args.result:
        if args.teleop:
            p.error("Replay cannot be teleoperated")
        from replay import Replay

        env = Replay(args.result, args.episode)
        env.seek(args.frame)
        outline, args.target = env.outline, env.target
    else:
        env = Env(args.robot, backend=args.backend)
        env.reset([args.seed], [args.target], args.prompt)
        outline = target_outline(args.target, args.seed)
    if args.out:
        render_scene(env, outline, args.camera, args.target, args.out)
    else:
        show_scene(env, outline, args.camera, args.target, args.teleop, args.seed)


if __name__ == "__main__":
    main()
