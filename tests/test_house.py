"""Physical regression for the selected house demonstration, not a success-rate study."""

import numpy as np

from env import Env
from eval import run_batch
from examples.house import Policy


def test_house_demo_assembles_through_robot_actions_and_holds_until_horizon():
    episodes = []
    run_batch(
        Env("panda"),
        Policy,
        [100076],
        12000,
        1,
        lambda row, trace: episodes.append((row, trace)),
        targets=["house"],
        max_inference_calls=12000,
    )
    row, trace = episodes[0]
    assert row["status"] == "completed", row["error"]
    assert row["policy_access"] == "oracle"
    assert row["success"] and row["iou"] >= 0.95
    assert row["final_hold_steps"] >= 25
    assert row["steps_executed"] == 12000
    assert trace["actions"].shape == (12000, 8)
    displacement = trace["pieces"][-1, :, :2] - trace["pieces"][0, :, :2]
    assert np.all(np.linalg.norm(displacement, axis=1) > 0.2)
