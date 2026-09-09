"""Regression tests for evaluation integrity, independent of policy competence."""

import json
from copy import deepcopy

import numpy as np
import pytest

from benchmark import (
    COMPARISON_FIELDS,
    CONTROL_SECONDS,
    PROTOCOL,
    SCHEMA_VERSION,
    PolicyDriver,
    compare_runs,
    score_trajectory,
    summarize,
    wilson_interval,
)
from env import Env
from eval import main, run_batch
from policy import Policy
from tangram import THICKNESS, goal, score_batch, square_solution
from tools.results import verify


def trajectory(passes):
    outline = goal(0)
    xy, yaw = square_solution(outline)
    poses = np.zeros((len(passes), 7, 7))
    poses[..., :2], poses[..., 2] = xy, THICKNESS / 2
    poses[..., 3], poses[..., 6] = np.cos(yaw / 2), np.sin(yaw / 2)
    poses[~np.asarray(passes), :, 0] += 1
    return {
        "pieces": poses,
        "piece_velocities": np.zeros((len(passes), 7, 6)),
        "goal": np.repeat(outline[None], len(passes), axis=0),
    }


def test_final_hold_excludes_reset_and_requires_full_consecutive_window():
    assert not score_trajectory(trajectory([True] * 25))["success"]
    held = score_trajectory(trajectory([False] + [True] * 25))
    assert held["success"] and held["first_success_seconds"] == 0.5
    assert held["pieces_in_goal"] == 7
    disturbed = score_trajectory(trajectory([False] + [True] * 25 + [False]))
    assert not disturbed["success"] and disturbed["first_success_seconds"] == 0.5
    recovered = score_trajectory(trajectory([False] + [True] * 25 + [False] + [True] * 24))
    assert not recovered["success"] and recovered["failure_reasons"] == ["insufficient_hold"]
    assert not score_trajectory(trajectory([True] * 30), completed=False)["success"]


@pytest.mark.parametrize("failure", ["nan", "quaternion", "goal", "shape"])
def test_scorer_rejects_invalid_evidence(failure):
    trace = trajectory([True])
    if failure == "nan":
        trace["pieces"][0, 0, 0] = np.nan
    elif failure == "quaternion":
        trace["pieces"][0, 0, 3:] = 0
    elif failure == "goal":
        trace["goal"][:] = 0
    else:
        trace["piece_velocities"] = np.zeros((1, 6, 6))
    with pytest.raises(ValueError):
        score_batch(trace["pieces"], trace["piece_velocities"], trace["goal"])


def test_chunk_validation_is_atomic_and_copies_policy_memory():
    env = Env()
    obs = env.reset([3])[0]
    action = np.r_[obs["qpos"][:7], 1.0]

    class Chunk:
        calls = 0
        output = np.stack([action, action])

        def act(self, observation):
            self.calls += 1
            observation["pieces"][:] = 100  # Must not alter scoring evidence.
            return self.output

    policy = Chunk()
    driver = PolicyDriver(policy, env.validate_actions, max_chunk=2)
    driver.act(obs)
    policy.output[:] = np.nan
    np.testing.assert_array_equal(driver.act(obs), action)
    assert policy.calls == 1 and obs["pieces"].max() < 2
    with pytest.raises(ValueError):
        driver.act(obs)
    policy.output = np.stack([action, action])
    policy.output[-1, -1] = 2
    with pytest.raises(ValueError, match="Gripper"):
        driver.act(obs)
    assert not driver.queue  # The valid first command never slips through.
    policy.output = np.stack([action] * 3)
    with pytest.raises(ValueError, match="chunk"):
        driver.act(obs)


@pytest.mark.parametrize("when", ["constructor", "act"])
def test_policy_error_does_not_drop_or_poison_other_worlds(when):
    class FailsOne(Policy):
        def __init__(self, robot, seed):
            if seed == 10 and when == "constructor":
                raise RuntimeError("bad checkpoint")
            super().__init__(robot, seed)
            self.seed = seed

        def act(self, obs):
            if self.seed == 10:
                return np.full(8, np.nan)
            return super().act(obs)

    rows = []
    run_batch(Env(num_envs=2), FailsOne, [10, 11], 3, 1, lambda row, trace: rows.append(row))
    assert [r["status"] for r in rows] == ["policy_error", "completed"]
    assert [r["steps_executed"] for r in rows] == [0, 3]
    assert not rows[0]["success"]
    assert summarize(rows)["policy_error_count"] == 1


def test_environment_error_aborts_and_records_unscorable_batch(monkeypatch):
    env = Env(num_envs=2)

    def fail(actions):
        raise RuntimeError("unstable simulation")

    monkeypatch.setattr(env, "step", fail)
    rows = []
    with pytest.raises(RuntimeError, match="unstable"):
        run_batch(env, Policy, [10, 11], 3, 1, lambda row, trace: rows.append(row))
    assert [r["status"] for r in rows] == ["environment_error"] * 2
    assert all("iou" not in r for r in rows)


def test_confidence_interval_includes_uncertainty_at_extremes():
    assert wilson_interval(0, 4) == pytest.approx([0, 0.4898908365])
    assert wilson_interval(4, 4) == pytest.approx([0.5101091635, 1])
    assert wilson_interval(0, 100)[1] < wilson_interval(0, 4)[1]


def make_result():
    rows = []
    for seed in range(4):
        row = score_trajectory(trajectory([False] + [bool(seed % 2)] * 25))
        rows.append({"seed": seed, "status": "completed", **row})
    return {
        **{field: "fixed" for field in COMPARISON_FIELDS},
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "completed",
        "episodes": rows,
        "requested_episodes": 4,
    }


def test_comparison_pairs_by_seed_and_counts_policy_errors():
    a = make_result()
    b = deepcopy(a)
    b["episodes"][0].update(status="policy_error", success=False)
    b["episodes"].reverse()
    result = compare_runs(a, b)
    assert result["ties"] == 4 and result["success_rate_difference_b_minus_a"] == 0
    b["episodes"][2]["success"] = False
    result = compare_runs(a, b)
    assert result["a_wins"] == 1 and result["success_rate_difference_b_minus_a"] == -0.25


@pytest.mark.parametrize("field", COMPARISON_FIELDS)
def test_comparison_rejects_different_contracts(field):
    a, b = make_result(), make_result()
    b[field] = "different"
    with pytest.raises(ValueError):
        compare_runs(a, b)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "different_seeds", "invalid"])
def test_comparison_does_not_silently_select_favorable_episodes(failure):
    a, b = make_result(), make_result()
    if failure == "missing":
        b["episodes"].pop()
    elif failure == "duplicate":
        b["episodes"][0]["seed"] = b["episodes"][1]["seed"]
    elif failure == "different_seeds":
        b["episodes"][0]["seed"] = 999
    else:
        b["status"] = "error"
    with pytest.raises(ValueError):
        compare_runs(a, b)


def test_cli_persists_traces_rescores_and_rejects_tampering(tmp_path):
    path = tmp_path / "run.json"
    assert main(["--episodes", "3", "--num-envs", "2", "--steps", "3", "--out", str(path)]) == 0
    run = verify(path)
    assert run["summary"]["episode_count"] == 3 and run["success_rate"] == 0
    with np.load(path.parent / run["episodes"][0]["trajectory"]) as trace:
        np.testing.assert_array_equal(trace["time"], np.arange(4) * CONTROL_SECONDS)
        assert trace["actions"].shape == trace["controls"].shape == (3, 8)
    assert len((tmp_path / "run.artifacts" / "episodes.jsonl").read_text().splitlines()) == 3
    with pytest.raises(SystemExit):
        main(["--out", str(path)])
    run["episodes"][0]["iou"] = 0.9
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Score mismatch"):
        verify(path)
    run["episodes"][0]["iou"] = 0.0
    path.write_text(json.dumps(run))
    with (path.parent / run["episodes"][0]["trajectory"]).open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(path)


def test_setup_failure_is_a_saved_invalid_run(tmp_path):
    candidate = tmp_path / "broken.py"
    candidate.write_text('raise RuntimeError("cannot load weights")\n')
    path = tmp_path / "run.json"
    assert main(["--policy", str(candidate), "--out", str(path)]) == 1
    run = json.loads(path.read_text())
    assert run["status"] == "error" and run["summary"] is None and run["success_rate"] is None
    assert run["error"]["message"] == "cannot load weights"
    with pytest.raises(ValueError, match="invalid"):
        verify(path)


def test_declared_source_mutation_invalidates_result(tmp_path):
    candidate = tmp_path / "changing.py"
    candidate.write_text(
        "from pathlib import Path\n"
        "from policy import Policy as Hold\n"
        "class Policy(Hold):\n"
        "    def act(self, obs):\n"
        '        Path(__file__).write_text("# changed during evaluation\\n")\n'
        "        return super().act(obs)\n"
    )
    path = tmp_path / "run.json"
    assert (
        main(["--policy", str(candidate), "--episodes", "1", "--steps", "1", "--out", str(path)])
        == 1
    )
    run = json.loads(path.read_text())
    assert run["status"] == "error" and run["summary"] is None
    assert "changed during evaluation" in run["error"]["message"]
    assert len(run["episodes"]) == 1


def test_keyboard_interrupt_preserves_prefix_and_has_no_aggregate(tmp_path, monkeypatch):
    original = Env.step

    def interrupt(self, actions):
        if self.steps == 2:
            raise KeyboardInterrupt()
        return original(self, actions)

    monkeypatch.setattr(Env, "step", interrupt)
    path = tmp_path / "run.json"
    assert main(["--episodes", "1", "--steps", "10", "--out", str(path)]) == 130
    run = json.loads(path.read_text())
    assert run["status"] == "interrupted" and run["summary"] is None
    row = run["episodes"][0]
    assert row["status"] == "interrupted" and row["steps_executed"] == 2
    with np.load(path.parent / row["trajectory"]) as trace:
        assert len(trace["pieces"]) == 3
        assert len(trace["actions"]) == 2


def test_policy_failure_can_be_verified_but_never_earns_success(tmp_path):
    candidate = tmp_path / "crash.py"
    candidate.write_text(
        "class Policy:\n"
        "    def __init__(self, robot, seed):\n"
        '        raise RuntimeError("bad weights")\n'
    )
    path = tmp_path / "run.json"
    assert (
        main(["--policy", str(candidate), "--episodes", "1", "--steps", "30", "--out", str(path)])
        == 0
    )
    run = verify(path)
    assert run["success_rate"] == 0
    assert run["summary"]["policy_error_count"] == 1
    assert run["summary"]["mean_final_iou_completed"] is None
