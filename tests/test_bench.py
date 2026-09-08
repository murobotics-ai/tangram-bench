"""Test the evaluator's failure modes, not whether a policy memorizes its code."""

import os

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from tangram import SIDE, THICKNESS, TILES, VERTICES, goal, polygons, score, square_solution


def solved(seed=0):
    outline = goal(seed)
    xy, yaw = square_solution(outline)
    poses = np.zeros((7, 7))
    poses[:, :2], poses[:, 2] = xy, THICKNESS / 2
    poses[:, 3], poses[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
    return outline, poses


def test_dissection_covers_square_without_overlap():
    tiles = [Polygon(p) for p in TILES]
    union = unary_union(tiles)
    assert union.area == pytest.approx(1)
    assert sum(p.area for p in tiles) == pytest.approx(union.area)
    assert sorted(p.area for p in tiles) == [0.0625, 0.0625, 0.125, 0.125, 0.125, 0.25, 0.25]


@pytest.mark.parametrize("seed", [0, 5, 100000, 200003])
def test_rotated_solution_reaches_inset_ceiling(seed):
    outline, poses = solved(seed)
    result = score(poses, np.zeros((7, 6)), outline)
    assert result["success"]
    assert result["iou"] == pytest.approx(sum(Polygon(v).area for v in VERTICES) / SIDE**2)


@pytest.mark.parametrize("failure", ["airborne", "moving", "stacked", "tilted", "outside"])
def test_invalid_physical_solutions_fail(failure):
    outline, poses = solved()
    vel = np.zeros((7, 6))
    if failure == "airborne":
        poses[:, 2] += 0.1
    elif failure == "moving":
        vel[0, 0] = 0.1
    elif failure == "stacked":
        poses[1] = poses[0]
        poses[1, 2] += THICKNESS
    elif failure == "tilted":
        poses[0, 3:] = [np.cos(0.2), np.sin(0.2), 0, 0]
    else:
        poses[0, 0] += 0.3
    assert not score(poses, vel, outline)["success"]


def test_identical_triangles_can_exchange_locations():
    # Tile 1 is congruent to tile 0 after a local +90 degree rotation.
    outline, poses = solved(4)
    old = poses.copy()
    _, yaw = square_solution(outline)
    poses[0, :2], poses[1, :2] = old[1, :2], old[0, :2]
    for i, angle in [(0, yaw + np.pi / 2), (1, yaw - np.pi / 2)]:
        poses[i, 3:] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    assert score(poses, np.zeros((7, 6)), outline)["success"]


@pytest.fixture
def env():
    from env import Env
    from tools.prepare import ASSETS

    if not ASSETS.exists():
        pytest.skip("Run uv run -m tools.prepare for physics tests")
    return Env()


def test_reset_is_reproducible_nonoverlapping_and_settled(env):
    a = env.reset([101])[0]
    b = env.reset([101])[0]
    np.testing.assert_array_equal(a["pieces"], b["pieces"])
    footprints = polygons(a["pieces"])
    for i, p in enumerate(footprints):
        assert all(p.intersection(q).area < 1e-8 for q in footprints[i + 1 :])
    assert np.all(np.abs(a["pieces"][:, 2] - THICKNESS / 2) < 0.001)
    assert not np.array_equal(a["pieces"], env.reset([102])[0]["pieces"])


def test_actions_cannot_teleport_or_inject_nan(env):
    from policy import Policy

    obs = env.reset([5])[0]
    p = Policy("panda", 5)
    action = p.act(obs)
    for bad in [np.zeros(3), np.full(8, np.nan), np.r_[action[:-1], 2.0], np.full(8, 100)]:
        with pytest.raises(ValueError):
            env.step([bad])
    initial = obs["pieces"].copy()
    for _ in range(20):
        obs = env.step([action])[0]
    assert np.max(np.abs(initial[:, :2] - obs["pieces"][:, :2])) < 0.001


@pytest.mark.parametrize("robot", ["panda", "piper"])
def test_robot_physics_smoke(robot):
    from env import Env
    from policy import Policy
    from tools.prepare import ASSETS

    if not ASSETS.exists():
        pytest.skip("Run uv run -m tools.prepare for physics tests")
    e = Env(robot)
    obs = e.reset([12])[0]
    p = Policy(robot, 12)
    for _ in range(20):
        obs = e.step([p.act(obs)])[0]
    assert np.isfinite(obs["pieces"]).all()
    assert np.all(abs(obs["pieces"][:, 2] - THICKNESS / 2) < 0.002)


@pytest.mark.skipif(
    os.environ.get("MU_BENCH_TEST_WARP") != "1", reason="Opt in to GPU integration tests"
)
def test_warp_batch_matches_cpu_at_rest_and_resets():
    from env import Env
    from policy import Policy

    cpu, gpu = Env(num_envs=2), Env(backend="warp", num_envs=2)
    for seeds in [[20, 21], [22, 23]]:
        a, b = cpu.reset(seeds), gpu.reset(seeds)
        policies = [Policy("panda", seed) for seed in seeds]
        for _ in range(10):
            actions = [p.act(o) for p, o in zip(policies, a)]
            a, b = cpu.step(actions), gpu.step(actions)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x["pieces"][:, :3], y["pieces"][:, :3], atol=0.001)
            np.testing.assert_allclose(x["qpos"], y["qpos"], atol=0.01)


def test_vectorized_projection_matches_full_rigid_transform():
    import mujoco

    from tangram import VERTICES

    rng = np.random.default_rng(2026)
    poses = rng.normal(size=(12, 7, 7))
    poses[..., 3:] /= np.linalg.norm(poses[..., 3:], axis=-1, keepdims=True)
    actual = polygons(poses)
    for b in range(12):
        for i, v in enumerate(VERTICES):
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, poses[b, i, 3:])
            xyz = np.column_stack((v, np.zeros(len(v)))) @ mat.reshape(3, 3).T
            expected = Polygon(xyz[:, :2] + poses[b, i, :2]).convex_hull
            assert actual[b, i].symmetric_difference(expected).area < 1e-12


def test_batch_metrics_keep_worlds_independent():
    from tangram import score_batch

    examples = [solved(seed) for seed in [4, 5, 6]]
    outlines = np.stack([x[0] for x in examples])
    poses = np.stack([x[1] for x in examples])
    velocities = np.zeros((3, 7, 6))
    poses[1, :, 2] += 0.1  # Same perfect projection, but suspended.
    poses[2, :, 0] += 0.3  # On the table, outside the target.
    metric = score_batch(poses, velocities, outlines)
    np.testing.assert_array_equal(metric["success"], [True, False, False])
    ceiling = sum(Polygon(v).area for v in VERTICES) / SIDE**2
    np.testing.assert_allclose(metric["iou"], [ceiling, ceiling, 0], atol=1e-12)
    np.testing.assert_array_equal(metric["on_table"], [True, False, True])


def test_upside_down_knob_fails_even_with_perfect_projection():
    outline, poses = solved()
    poses[:, 3:] = [0, 1, 0, 0]
    assert not score(poses, np.zeros((7, 6)), outline)["flat"]


@pytest.mark.parametrize("robot", ["panda", "piper"])
def test_knob_geometry_mass_and_clearance(robot):
    from env import Env
    from tangram import INSET, KNOB_HEIGHT, KNOB_WIDTH, NOMINAL_VERTICES, knob_yaw, rotation

    e = Env(robot)
    for i, (v, nominal) in enumerate(zip(VERTICES, NOMINAL_VERTICES)):
        slab = Polygon(v)
        expected = Polygon(nominal).buffer(-INSET, join_style="mitre")
        assert slab.symmetric_difference(expected).area < 1e-12
        knob = e.model.geom(f"knob{i}")
        np.testing.assert_allclose(knob.size, [KNOB_WIDTH / 2, KNOB_WIDTH / 2, KNOB_HEIGHT / 2])
        np.testing.assert_allclose(knob.pos, [0, 0, THICKNESS / 2 + KNOB_HEIGHT / 2])
        assert knob.bodyid == e.model.body(f"piece{i}").id
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * KNOB_WIDTH / 2
        assert slab.contains(Polygon(corners @ rotation(knob_yaw(v)).T))
        mass = (slab.area * THICKNESS + KNOB_WIDTH**2 * KNOB_HEIGHT) * 700
        body = e.model.body(f"piece{i}")
        assert body.mass[0] == pytest.approx(mass, rel=1e-6)
        assert body.ipos[2] > 0
        assert np.all(body.inertia > 0)


@pytest.mark.parametrize("robot", ["panda", "piper"])
def test_gripper_squeezes_a_knob_hard_enough_to_lift_any_piece(robot):
    """Position-servo grippers squeeze with stiffness x gap; a 2 cm knob leaves a 1 cm gap."""
    from env import make_model
    from tangram import KNOB_WIDTH

    m = make_model(robot)
    i = m.nu - 1
    gap = m.actuator_ctrlrange[i, 1] * (0.04 / 255 if robot == "panda" else 1) - KNOB_WIDTH / 2
    squeeze = min(
        -m.actuator_biasprm[i, 1] * (2 * gap if robot == "panda" else gap),
        m.actuator_forcerange[i, 1],
    )
    heaviest = max(m.body(f"piece{k}").mass for k in range(7)) * 9.81
    # Two pads at friction 1.0 must hold the heaviest piece with a 3x safety factor.
    assert 2 * squeeze >= 3 * heaviest
