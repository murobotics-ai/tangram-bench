"""Scene/tooling checks only: no policy imports, actions, scoring or evaluation."""

import mujoco
import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

import view
from env import Env
from tangram import INSET, SIDE, THICKNESS, VERTICES, goal


@pytest.mark.parametrize("robot,parent", [("panda", "hand"), ("piper", "link6")])
def test_cameras_have_correct_mounts_and_follow_only_the_wrist(robot, parent):
    env = Env(robot)
    env.reset([100000])
    m, d = env.model, env.data[0]
    wrist, context = m.camera("wrist").id, m.camera("context").id
    assert m.cam_bodyid[wrist] == m.body(parent).id
    assert m.cam_bodyid[context] == 0
    assert m.cam_bodyid[m.camera("top").id] == 0
    initial_wrist, initial_context = d.cam_xpos[wrist].copy(), d.cam_xpos[context].copy()
    d.qpos[0] += 0.2
    mujoco.mj_forward(m, d)
    assert not np.allclose(d.cam_xpos[wrist], initial_wrist)
    np.testing.assert_array_equal(d.cam_xpos[context], initial_context)
    body = m.body(parent).id
    local_position = d.xmat[body].reshape(3, 3).T @ (d.cam_xpos[wrist] - d.xpos[body])
    np.testing.assert_allclose(local_position, m.cam_pos[wrist], atol=1e-12)


@pytest.mark.parametrize("robot", ["panda", "piper"])
def test_packed_square_layout_and_context_framing(robot):
    env = Env(robot)
    for seed in [*range(100), 100000, 200000]:
        obs = env.reset([seed])[0]
        outlines = []
        for v, pose in zip(VERTICES, obs["pieces"]):
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, pose[3:])
            corners = np.column_stack((v, np.zeros(len(v)))) @ mat.reshape(3, 3).T + pose[:3]
            outlines.append(Polygon(corners[:, :2]))
            assert corners[:, 1].max() < -0.02  # Source stays on the -y side.
        assert all(p.distance(Polygon(obs["goal"])) > 0.015 for p in outlines)
        packed = unary_union(outlines)
        rectangle = packed.minimum_rotated_rectangle
        assert packed.area == pytest.approx(sum(Polygon(v).area for v in VERTICES), abs=1e-7)
        assert packed.area / rectangle.area > 0.97
        sides = np.linalg.norm(np.diff(np.array(rectangle.exterior.coords), axis=0), axis=1)
        np.testing.assert_allclose(sides, SIDE - 2 * INSET, atol=1e-5)
        assert 0.28 - 1e-5 <= packed.centroid.x <= 0.36 + 1e-5
        assert -0.30 - 1e-5 <= packed.centroid.y <= -0.24 + 1e-5
        for i, p in enumerate(outlines):
            assert all(p.distance(q) >= 2 * INSET - 1e-5 for q in outlines[i + 1 :])
        np.testing.assert_allclose(obs["pieces"][:, 2], THICKNESS / 2, atol=0.001)
        np.testing.assert_array_equal(obs["goal"], goal(seed))
        # Camera frustum contains robot body origins, pieces and all target corners.
        d, m = env.data[0], env.model
        camera = m.camera("context").id
        points = np.vstack((d.xpos, obs["pieces"][:, :3], np.c_[obs["goal"], np.zeros(4)]))
        local = (points - d.cam_xpos[camera]) @ d.cam_xmat[camera].reshape(3, 3)
        depth = -local[:, 2]
        tan = np.tan(np.deg2rad(m.cam_fovy[camera] / 2))
        assert np.all(depth > 0)
        assert np.all(abs(local[:, 1]) < depth * tan)
        assert np.all(abs(local[:, 0]) < depth * tan * 960 / 720)
    assert not np.allclose(env.reset([0])[0]["pieces"], env.reset([1])[0]["pieces"])
    a = env.reset([100000])[0]
    b = env.reset([100000])[0]
    np.testing.assert_array_equal(a["pieces"], b["pieces"])
    assert set(a) == {
        "robot",
        "prompt",
        "target",
        "time",
        "qpos",
        "qvel",
        "tcp_pos",
        "tcp_mat",
        "pieces",
        "piece_velocities",
        "goal",
    }


def test_undefined_silhouettes_are_not_selectable():
    assert set(view.TARGETS) == {"square", "rectangle", "house", "cat"}
    with pytest.raises(ValueError, match="not defined"):
        view.target_outline("horse", 0)
    assert "pixels" in view.panel_text("square", "context")


@pytest.mark.parametrize("size", [(1440, 960), (720, 600), (1920, 1080)])
def test_panels_fit_without_overlap(size):
    main, panels = view.layout(*size)
    rectangles = [main, *panels]
    for i, r in enumerate(rectangles):
        assert r.width > 0 and r.height > 28
        assert 0 <= r.left and r.left + r.width <= size[0]
        assert 0 <= r.bottom and r.bottom + r.height <= size[1]
        for q in rectangles[i + 1 :]:
            assert (
                r.left + r.width <= q.left
                or q.left + q.width <= r.left
                or r.bottom + r.height <= q.bottom
                or q.bottom + q.height <= r.bottom
            )


def test_goal_panel_keeps_orientation_and_is_translation_invariant():
    outline = goal(42)
    image = view.goal_image(tuple(map(tuple, outline)), 360, 240)
    shifted = view.goal_image(tuple(map(tuple, outline + [1, 2])), 360, 240)
    np.testing.assert_array_equal(image, shifted)
    assert image.shape == (240, 360, 3)
    assert np.any(image < 30) and np.any(image > 240)
    assert not np.array_equal(image, view.goal_image(tuple(map(tuple, goal(43))), 360, 240))


@pytest.mark.parametrize("robot", ["panda", "piper"])
def test_teleop_moves_tcp_and_gripper_through_physics(robot):
    from teleop import Teleop

    env = Env(robot)
    obs = env.reset([100000])[0]
    control = Teleop(env)
    initial = env.data[0].qpos.copy()
    action = control.action()
    np.testing.assert_array_equal(env.data[0].qpos, initial)
    assert action.shape == (env.narm + 1,)
    for _ in range(40):
        control.step([0, 0, 1], [0, 0, 0])
    moved = env.observe()[0]
    assert moved["tcp_pos"][2] > obs["tcp_pos"][2] + 0.03
    assert control.error < 0.025
    # Pieces only move through contact, not through the target/IK scratch state.
    np.testing.assert_allclose(moved["pieces"][:, :3], obs["pieces"][:, :3], atol=1e-4)
    for _ in range(60):
        control.step([0, 0, 0], [0, 0, 0], -1)
    closed = abs(env.data[0].qpos[env.narm : env.narm + 2]).sum()
    for _ in range(60):
        control.step([0, 0, 0], [0, 0, 0], 1)
    opened = abs(env.data[0].qpos[env.narm : env.narm + 2]).sum()
    assert closed < 0.005 and opened > 0.06
    before_rotation = env.observe()[0]["tcp_mat"]
    for _ in range(25):
        control.step([0, 0, 0], [0, 0, 1])
    assert np.linalg.norm(env.observe()[0]["tcp_mat"] - before_rotation) > 0.05
    control.paused = True
    before = env.data[0].qpos.copy()
    steps = env.steps
    control.step([1, 1, 1], [1, 1, 1], -1)
    np.testing.assert_array_equal(env.data[0].qpos, before)
    assert env.steps == steps


def test_teleop_bounds_targets_and_reset_discards_commands():
    from teleop import Teleop

    env = Env()
    env.reset([42])
    control = Teleop(env)
    for _ in range(100):
        control.move([1, 1, 1], [0, 0, 0], -1)
    assert np.linalg.norm(control.position - env.data[0].site_xpos[env.tcp]) <= 0.040001
    action = control.action()
    assert np.isfinite(action).all()
    assert np.all(action[:-1] >= env.limits[:-1, 0])
    assert np.all(action[:-1] <= env.limits[:-1, 1])
    assert action[-1] == 0
    env.reset([42])
    reset = Teleop(env)
    np.testing.assert_allclose(reset.position, env.observe()[0]["tcp_pos"])
    assert reset.grip == 1 and env.steps == 0


def test_rotation_error_detects_half_turn_and_limits_orientation_windup():
    from teleop import Teleop, orientation_error, rotated

    half_turn = rotated(np.eye(3), [np.pi, 0, 0])
    assert np.linalg.norm(orientation_error(half_turn, np.eye(3))) == pytest.approx(np.pi)
    env = Env()
    env.reset([42])
    control = Teleop(env)
    for _ in range(300):
        control.move([0, 0, 0], [1, 1, 1])
    current = env.observe()[0]["tcp_mat"]
    assert np.linalg.norm(orientation_error(control.orientation, current)) <= 0.350001


@pytest.mark.parametrize("robot", ["panda", "piper"])
def test_ik_errors_limits_and_unreachable_targets(robot):
    from teleop import Teleop, orientation_error

    env = Env(robot)
    env.reset([42])
    control = Teleop(env)
    control.position += [0, 0, 0.01]
    action = control.action()
    assert control.ik_error[0] < 0.001
    assert control.ik_error[1] < np.deg2rad(1)
    # Residuals describe the returned command, not the previous iteration.
    d = mujoco.MjData(env.model)
    d.qpos[:] = env.data[0].qpos
    d.qpos[: env.narm] = action[:-1]
    mujoco.mj_forward(env.model, d)
    np.testing.assert_allclose(
        control.ik_error,
        [
            np.linalg.norm(control.position - d.site_xpos[env.tcp]),
            np.linalg.norm(
                orientation_error(control.orientation, d.site_xmat[env.tcp].reshape(3, 3))
            ),
        ],
        atol=1e-10,
    )
    for q in [env.limits[:-1, 0] + 1e-5, env.limits[:-1, 1] - 1e-5]:
        env.data[0].qpos[: env.narm] = q
        mujoco.mj_forward(env.model, env.data[0])
        control = Teleop(env)
        control.position += [10, 10, 10]
        action = control.action()
        assert np.isfinite(action).all()
        assert np.all(action[:-1] >= env.limits[:-1, 0])
        assert np.all(action[:-1] <= env.limits[:-1, 1])
        assert control.ik_error[0] > 1
        assert "NOT CONVERGED" in control.text()


def test_panda_nullspace_prefers_posture_without_losing_tcp():
    from teleop import Teleop

    env = Env()
    env.reset([42])
    control = Teleop(env)
    d = env.data[0]
    jp, jr = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, d, jp, jr, env.tcp)
    _, _, vt = np.linalg.svd(np.vstack((jp[:, :7], 0.2 * jr[:, :7])), full_matrices=True)
    control.posture += vt[-1] * 0.25
    before = np.linalg.norm(d.qpos[:7] - control.posture)
    action = control.action()
    assert np.linalg.norm(action[:7] - control.posture) < before * 0.95
    assert control.ik_error[0] < 0.001
    assert control.ik_error[1] < np.deg2rad(1)


def test_piper_singularity_is_reported_without_invalid_action():
    from teleop import Teleop

    env = Env("piper")
    env.reset([42])
    env.data[0].qpos[:6] = 0
    mujoco.mj_forward(env.model, env.data[0])
    control = Teleop(env)
    control.position += [0.001, 0.001, 0.001]
    action = control.action()
    assert control.singular
    assert np.isfinite(action).all()
    assert "near singular" in control.text()
