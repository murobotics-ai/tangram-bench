"""Scene inspection, teleoperation and replay with goal, top and wrist panels.

Static by default; --teleop enables manual physics, and --record DIR saves each
teleoperated episode as a demonstration in the tools/collect.py format.
--result replays an evaluation episode and --demo a recorded demonstration; both
show the task prompt and the per-step language annotation as it changes.
Never executes a policy.
"""

import argparse
import json
import textwrap
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import shapely

from env import Env, draw_goal
from shapes import TARGETS, canonical
from tangram import goal

CAMERAS = ("context", "wrist", "top")
PROMPT_HEIGHT = 84
RECORD_FPS = 10


def target_outline(name, seed):
    if name not in TARGETS:
        raise ValueError(f"Target {name!r} is not defined; available: {', '.join(TARGETS)}")
    return goal(seed, name)


def panel_text(selected, camera):
    return (
        f"TANGRAM-BENCH  |  {selected.title()}\n"
        f"Main: {camera}  |  1 context / 2 wrist / 3 top\n"
        "Policy input: state, or top/wrist pixels with --obs pixels\n"
        "Context camera: inspection only"
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


def set_camera(camera, model, name):
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = model.camera(name).id


def draw_frame(
    env, scene, context, width, height, camera, target, outline, control=None, session=None
):
    """All cameras see the same state and the visual-only target on the table."""
    main, panels = layout(width, height)
    mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0.055, 0.065, 0.08, 1)
    # Language caption: centred over the scene, drawn with the overlay's own box so it
    # reads as one label rather than a panel within a panel.
    caption = mujoco.MjrRect(0, height - PROMPT_HEIGHT + 10, width, PROMPT_HEIGHT - 20)
    columns = max(30, (min(width, 1100) - 60) // 11)
    lines = textwrap.fill("TASK  " + env.prompt, width=columns)
    now = env.label() if hasattr(env, "label") else ""
    if now:
        # Indent the subtask under the task so the two levels read as a hierarchy.
        indent = " " * 6
        lines += "\n" + textwrap.fill(
            now.replace(": ", "  ", 1),
            width=columns,
            initial_indent=indent,
            subsequent_indent=indent,
        )
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOP, caption, lines, "", context
    )
    viewport = image_rect(panels[0])
    # The panel shows the figure upright; its real pose on the table is in every camera.
    pixels = goal_image(tuple(map(tuple, canonical(target)[0])), viewport.width, viewport.height)
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
        f"GOAL  /  {target.title()}",
        "TOP CAMERA  (policy input)",
        "WRIST CAMERA  (policy input)",
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
        else (
            f"TANGRAM-BENCH | {target.title()} | 1/2/3: main camera"
            + (
                f" | REC {len(session.recorder.frames['time'])} frames"
                + (" SOLVED" if getattr(session, "held", False) else "")
                if session is not None
                else ""
            )
            + "\n"
            + control.text()
        ),
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


class Session:
    """Teleoperation recording: one episode per reset, saved like tools/collect.py."""

    def __init__(self, env, seed, target, outline, directory):
        from tools.collect import Recorder

        self.env, self.seed, self.target, self.outline = env, seed, target, outline
        self.directory = directory
        self.recorder = Recorder(env, seed, RECORD_FPS, env.prompt)
        self.tick = self.streak = 0

    def before(self):
        self.recorder.observe(self.env.observe()[0], self.tick)

    def after(self, action):
        from benchmark import HOLD_STEPS
        from tangram import score

        self.recorder.act(action, self.tick)
        self.tick += 1
        obs = self.env.observe()[0]
        solved = score(obs["pieces"], obs["piece_velocities"], obs["goal"])["success"]
        self.streak = self.streak + 1 if solved else 0
        self.held = getattr(self, "held", False) or self.streak >= HOLD_STEPS

    def finish(self, action):
        self.recorder.flush(action)
        if not self.recorder.frames["time"]:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.directory / f"episode-{self.seed}-{stamp}.npz"
        held = getattr(self, "held", False)
        row = self.recorder.save(
            path, self.target, held, "completed", None, self.tick, self.outline
        )
        row["source"] = "teleop"
        with (self.directory / "index.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return row


def show_scene(env, outline, camera, target, teleop=False, seed=100000, record=None):
    import glfw

    from teleop import CONTROL_DT, Teleop

    control = Teleop(env) if teleop else None
    session = Session(env, seed, target, outline, record) if record else None
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
        # Launched from a terminal the window can land behind the editor: ask for the front.
        glfw.focus_window(window)
        glfw.request_window_attention(window)
        context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        scene = mujoco.MjvScene(env.model, maxgeom=1000)

        def key_callback(window, key, scancode, action, mods):
            nonlocal camera, control, paused, session
            if action != glfw.PRESS:
                return
            if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
                camera = CAMERAS[key - glfw.KEY_1]
            elif control is not None and key == glfw.KEY_P:
                control.paused = not control.paused
            elif control is not None and key == glfw.KEY_O:
                if session is not None:
                    session.finish(control.last_action)
                env.reset([seed], [target], env.prompt)  # SpaceMouse right button
                control = Teleop(env)
                if session is not None:
                    session = Session(env, seed, target, outline, record)
            elif playback and key == glfw.KEY_SPACE:
                paused = not paused
            elif playback and key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                second = round(1 / env.frame_seconds)
                env.seek(env.steps + (second if key == glfw.KEY_RIGHT else -second))
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
                    if session is not None and not control.paused:
                        session.before()  # Renders frames in the recorder's own GL context.
                        glfw.make_context_current(window)
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
                    if session is not None and not control.paused:
                        session.after(control.last_action)
                    accumulator -= CONTROL_DT
            elif playback and not paused:
                while accumulator >= env.frame_seconds:
                    env.seek(env.steps + 1)
                    accumulator -= env.frame_seconds
                if env.steps == env.frames - 1:
                    paused = True
            else:
                accumulator = 0

            width, height = glfw.get_framebuffer_size(window)
            if width >= 720 and height >= 600:
                draw_frame(
                    env, scene, context, width, height, camera, target, outline, control, session
                )
                glfw.swap_buffers(window)
            glfw.wait_events_timeout(1 / 30)
    finally:
        if session is not None and control is not None:
            session.finish(control.last_action)
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
    p.add_argument("--prompt", default=None, help="Override the per-figure task text")
    p.add_argument(
        "--camera",
        choices=CAMERAS,
        default="context",
        help="Main view; top/wrist panels remain visible",
    )
    p.add_argument("--teleop", action="store_true", help="Manual TCP and gripper control")
    p.add_argument(
        "--record",
        type=Path,
        help="With --teleop: save each episode (reset or exit ends one) to this folder",
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--result", type=Path, help="Replay a saved evaluation")
    p.add_argument("--demo", type=Path, help="Replay a recorded demonstration (episode-*.npz)")
    p.add_argument("--episode", type=int, default=0, help="Zero-based episode index in --result")
    p.add_argument("--frame", type=int, default=0, help="Initial replay frame")
    args = p.parse_args()
    if args.teleop and args.out:
        p.error("--teleop is interactive; use --out without --teleop for a static PNG")
    if args.record and not args.teleop:
        p.error("--record needs --teleop")
    if args.result or args.demo:
        if args.teleop:
            p.error("Replay cannot be teleoperated")
        from replay import Demo, Replay

        env = Demo(args.demo) if args.demo else Replay(args.result, args.episode)
        env.seek(args.frame)
        outline, args.target = env.outline, env.target
    else:
        env = Env(args.robot, backend=args.backend, pixels=bool(args.record))
        env.reset([args.seed], [args.target], args.prompt)
        outline = target_outline(args.target, args.seed)
    if args.out:
        render_scene(env, outline, args.camera, args.target, args.out)
    else:
        show_scene(env, outline, args.camera, args.target, args.teleop, args.seed, args.record)


if __name__ == "__main__":
    main()
