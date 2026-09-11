"""Reference controller: a short physical smoke test, and the full assembly under TANGRAM_SLOW."""

import os

import numpy as np
import pytest

from env import Env
from eval import run_batch
from examples.oracle import Policy, placement_error, region
from shapes import solution


def test_region_words_follow_the_silhouette_frame():
    outline = np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.2], [0.0, 0.2]])
    assert region([0.2, 0.1], outline, 0.0) == "center"
    assert region([0.38, 0.19], outline, 0.0) == "top-right"
    assert region([0.02, 0.1], outline, 0.0) == "left"
    turned = outline @ np.array([[0, -1], [1, 0]]).T  # the same box rotated a quarter turn
    assert region(turned[2] - [0.01, 0.01], turned, np.pi / 2) == "top-right"


def test_placement_error_is_zero_at_the_certificate_and_wraps_yaw():
    poses = solution("house", 5)
    assert placement_error(poses[0], poses[0]) == (0.0, 0.0)
    flat = np.array([0.1, 0.2, 0.0025, 1, 0, 0, 0])
    turned = flat.copy()
    turned[3:] = [np.cos(np.pi - 0.05), 0, 0, np.sin(np.pi - 0.05)]  # yaw 2pi - 0.1
    assert placement_error(turned, flat)[1] == pytest.approx(0.1, abs=1e-6)


def test_oracle_lifts_and_carries_the_first_piece_through_contact():
    env = Env("panda")
    obs = env.reset([100076], ["house"])[0]
    policy = Policy("panda", 100076)
    start = obs["pieces"][policy.order[0], :3].copy()
    lifted = False
    for _ in range(900):
        obs = env.step([policy.act(obs)])[0]
        lifted |= obs["pieces"][policy.order[0], 2] > 0.05
    assert lifted, "first piece never left the table"
    assert "orange large triangle" in policy.subtask and "house" in policy.subtask
    assert len(policy.plan) == 7 and policy.step == 1
    assert policy.plan[0].startswith("Place the orange large triangle at the")
    assert policy.phase in ("transit", "transfer", "hover", "lower", "release", "retreat")
    assert np.linalg.norm(obs["pieces"][policy.order[0], :2] - start[:2]) > 0.05
    assert policy.high in (0.22, 0.14)


@pytest.mark.skipif(
    not os.environ.get("TANGRAM_SLOW"), reason="set TANGRAM_SLOW=1 (about a minute)"
)
@pytest.mark.parametrize("target,seed", [("house", 100076), ("rectangle", 3)])
def test_oracle_assembles_and_holds_until_horizon(target, seed):
    episodes = []
    run_batch(
        Env("panda"),
        Policy,
        [seed],
        12000,
        1,
        lambda row, trace: episodes.append((row, trace)),
        targets=[target],
        max_inference_calls=12000,
    )
    row, trace = episodes[0]
    assert row["status"] == "completed", row["error"]
    assert row["policy_access"] == "oracle"
    assert row["success"] and row["iou"] >= 0.95
    assert row["final_hold_steps"] >= 25
    displacement = trace["pieces"][-1, :, :2] - trace["pieces"][0, :, :2]
    assert np.all(np.linalg.norm(displacement, axis=1) > 0.2)
