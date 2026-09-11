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
@pytest.mark.parametrize("target,seed", [("house", 0), ("rectangle", 3)])
def test_oracle_assembles_and_holds_until_horizon(target, seed):
    episodes = []
    run_batch(
        Env("panda"),
        Policy,
        [seed],
        15000,
        1,
        lambda row, trace: episodes.append((row, trace)),
        targets=[target],
        max_inference_calls=15000,
    )
    row, trace = episodes[0]
    assert row["status"] == "completed", row["error"]
    assert row["policy_access"] == "oracle"
    from tangram import IOU_THRESHOLD

    assert row["success"] and row["iou"] >= IOU_THRESHOLD
    assert row["final_hold_steps"] >= 25
    displacement = trace["pieces"][-1, :, :2] - trace["pieces"][0, :, :2]
    assert np.all(np.linalg.norm(displacement, axis=1) > 0.2)


def test_approach_glides_in_joint_space_and_wound_up_arm_switches_grasp():
    env = Env("panda")
    obs = env.reset([3], ["house"])[0]
    policy = Policy("panda", 3)
    actions = [policy.act(obs)]
    for _ in range(60):
        obs = env.step([actions[-1]])[0]
        actions.append(policy.act(obs))
    steps = np.abs(np.diff(np.array(actions)[:, :7], axis=0)).max(axis=1)
    assert policy.phase == "approach" and steps.max() < 0.03, "approach must ramp, not jump"
    chosen = (policy.high, policy.turn)
    # A wound-up wrist while carrying lets go at once and bans that grasp for the piece.
    policy.phase = "transit"
    tilted = dict(obs)
    tilted["tcp_mat"] = np.eye(3)  # tool axis pointing up
    policy.act(tilted)
    assert policy.phase == "release" and chosen in policy.banned and policy.regrasp
    policy.banned = {(h, t) for h in (0.22, 0.14) for t in range(4)} - {(0.14, 2)}
    policy.choose_grasp(obs)
    assert (policy.high, policy.turn) == (0.14, 2), "only the unbanned grasp remains"


def test_ik_keeps_a_joint_pinned_at_its_limit_instead_of_leaking_into_the_task():
    from teleop import Teleop

    env = Env("panda")
    env.reset([0], ["square"])
    control = Teleop(env)
    upper = env.limits[6, 1]
    env.data[0].qpos[6] = upper  # wrist roll against its stop
    import mujoco

    mujoco.mj_forward(env.model, env.data[0])
    control.position = env.data[0].site_xpos[env.tcp].copy()
    control.orientation = env.data[0].site_xmat[env.tcp].reshape(3, 3).copy()
    control.posture = env.data[0].qpos[:7].copy()
    control.posture[6] = upper + 1.0  # the posture pull would push past the limit
    command = control.action()
    assert upper - 0.031 <= command[6] <= upper, "held at the stop, inside the soft-limit margin"
    assert control.ik_error[0] < 1e-3, "no stray TCP motion from the clipped pull"


def test_stalled_arm_replans_within_seconds_instead_of_the_hard_timeout():
    env = Env("panda")
    obs = env.reset([4], ["house"])[0]
    policy = Policy("panda", 4)
    # The world never advances: the same observation every tick, so the hand never
    # gets closer to its setpoint. The controller must let go and re-plan quickly.
    for tick in range(600):
        policy.act(obs)
        if policy.phase == "release":
            break
    assert policy.phase == "release" and tick < 500, "stall not detected within the glide + 3 s"
    assert policy.retries == 1 and policy.regrasp and len(policy.banned) == 1


def test_stalled_retreat_moves_on_and_a_disturbed_piece_gets_repaired():
    env = Env("panda")
    obs = env.reset([1], ["square"])[0]
    policy = Policy("panda", 1)
    policy.act(obs)  # plans the first piece
    # A retreat that makes no progress must hand over to the next grasp, not hang.
    policy.transition("retreat", obs)
    policy.release_position, policy.release_R = obs["tcp_pos"].copy(), obs["tcp_mat"].copy()
    policy.index = 3
    for _ in range(400):
        policy.act(obs)
        if policy.phase not in ("retreat", "start"):
            break
    assert policy.phase == "approach" and policy.index == 4, "retreat stall must advance"
    # After the first pass, a piece off its certificate pose is planned again.
    policy.index = 7
    policy.transition("start", obs)
    solved = obs.copy()
    solved["pieces"] = np.array(policy.targets, dtype=float)
    policy.act(solved)
    assert policy.phase == "done" and policy.repairs == 0
    disturbed = obs.copy()
    disturbed["pieces"] = np.array(policy.targets, dtype=float)
    disturbed["pieces"][4, 0] += 0.15  # brown small triangle 15 cm out: the scorer rejects it
    disturbed["piece_velocities"] = np.zeros((7, 6))
    policy.phase = "start"
    policy.act(disturbed)
    assert policy.phase == "approach" and policy.piece == 4 and policy.repairs == 1


def test_accepted_assembly_is_never_repaired_and_high_or_tilted_pieces_are():
    env = Env("panda")
    obs = env.reset([2], ["square"])[0]
    policy = Policy("panda", 2)
    policy.act(obs)
    policy.index = 7
    # Every piece 3 mm off its certificate pose: the scorer accepts it, so done.
    shifted = dict(obs)
    shifted["pieces"] = np.array(policy.targets, dtype=float)
    shifted["pieces"][:, 0] += 0.003
    shifted["piece_velocities"] = np.zeros((7, 6))
    policy.transition("start", shifted)
    policy.act(shifted)
    assert policy.phase == "done" and policy.repairs == 0
    # A piece standing 10 mm high is flagged even at the exact certificate pose.
    high = dict(obs)
    high["pieces"] = np.array(policy.targets, dtype=float)
    high["pieces"][2, 2] += 0.010
    assert policy.misplaced(high) == 2
    tilted = dict(obs)
    tilted["pieces"] = np.array(policy.targets, dtype=float)
    tilted["pieces"][5, 3:] = [np.cos(0.1), np.sin(0.1), 0, 0]  # 11.5 degrees about x
    assert policy.misplaced(tilted) == 5


def test_collision_cost_counts_hand_and_finger_contacts():
    import mujoco

    env = Env("panda")
    env.reset([0], ["square"])
    policy = Policy("panda", 0)
    policy.act(env.observe()[0])
    rng = np.random.default_rng(0)
    limits = policy.model.actuator_ctrlrange[:7]
    found = 0
    gripper = {"hand", "left_finger", "right_finger"}
    for _ in range(300):
        policy.data.qpos[:7] = rng.uniform(limits[:, 0], limits[:, 1])
        mujoco.mj_forward(policy.model, policy.data)
        pairs = []
        for i in range(policy.data.ncon):
            c = policy.data.contact[i]
            pairs.append(
                {
                    policy.model.body(policy.model.geom_bodyid[c.geom1]).name,
                    policy.model.body(policy.model.geom_bodyid[c.geom2]).name,
                }
            )
        # A hand or finger pressing on the table or on an arm link is a clash;
        # pieces (at their model defaults here) and the finger pair are not.
        if any(
            p & gripper
            and p <= gripper | {"world"} | {f"link{k}" for k in range(8)}
            and p != {"left_finger", "right_finger"}
            for p in pairs
        ):
            found += 1
            assert policy.clashes() > 0, "hand contact ignored by the collision cost"
    assert found > 0, "no hand contact sampled; widen the search"


def test_released_piece_inside_the_scorer_tolerance_is_not_picked_up_again():
    env = Env("panda")
    obs = env.reset([2], ["rectangle"])[0]
    policy = Policy("panda", 2)
    policy.act(obs)
    policy.index, policy.piece = 6, 5
    resting = dict(obs)
    resting["pieces"] = np.array(policy.targets, dtype=float)
    resting["pieces"][5, 2] += 0.003  # 3 mm up on a neighbour's edge: accepted by the scorer
    resting["piece_velocities"] = np.zeros((7, 6))
    policy.transition("retreat", resting)
    policy.finish_retreat(resting)
    assert policy.index == 7 and policy.retries == 0, "accepted assembly must not be re-grasped"
    policy.act(resting)
    assert policy.phase == "done"


def test_failed_carry_lowers_the_piece_before_letting_go():
    env = Env("panda")
    obs = env.reset([7], ["square"])[0]
    policy = Policy("panda", 7)
    policy.act(obs)
    policy.phase, policy.high = "transit", 0.22
    high = dict(obs)
    high["tcp_mat"] = np.eye(3)  # hand pointing up: the safety rule fires
    high["pieces"] = np.array(obs["pieces"], dtype=float)
    high["pieces"][policy.piece, 2] = 0.25  # the slab is 25 cm up, in the fingers
    policy.act(high)
    assert policy.phase == "abort", "a slab held high must be lowered, not dropped"
    assert policy.abort_position[2] == pytest.approx(obs["tcp_pos"][2] - (0.25 - 0.012))
    policy.act(high)  # the annotation follows on the next tick
    assert "lower it to the table" in policy.subtask
    # Once the setpoint is reached (or the descent stalls) the fingers open.
    for _ in range(600):
        policy.act(high)
        if policy.phase == "release":
            break
    assert policy.phase == "release"


def test_spent_retries_abandon_the_piece_instead_of_ending_the_episode():
    env = Env("panda")
    obs = env.reset([8], ["rectangle"])[0]
    policy = Policy("panda", 8)
    policy.act(obs)
    policy.index, policy.retries = 2, 3
    still = dict(obs)
    still["piece_velocities"] = np.zeros((7, 6))
    policy.transition("descend", still)
    policy.recover(still, "Grasp failed")  # fourth failure on this piece
    assert policy.abandon and policy.phase == "release", "no exception, let go and move on"
    policy.transition("retreat", still)
    policy.finish_retreat(still)
    assert policy.index == 3 and policy.retries == 0 and not policy.abandon


def test_waypoint_phase_that_stalls_within_two_centimetres_moves_on():
    env = Env("panda")
    obs = env.reset([9], ["cat"])[0]
    policy = Policy("panda", 9)
    policy.act(obs)
    # Hold the observation still with the hand 7 mm short of the transit setpoint:
    # no progress, but close enough for the next phase to take over.
    policy.phase, policy.high = "transit", 0.22
    policy.transition("transit", obs)
    from examples.oracle import VIA, via_needed

    target = policy.targets[policy.piece]
    waypoint = VIA if via_needed(policy.source, target[:2]) else target[:2]
    frozen = dict(obs)
    frozen["tcp_pos"] = np.r_[waypoint + [0.005, 0.005], 0.235]  # 7 mm short of the waypoint
    frozen["pieces"] = np.array(obs["pieces"], dtype=float)
    frozen["pieces"][policy.piece, :3] = [*frozen["tcp_pos"][:2], 0.20]  # slab in the fingers
    for _ in range(700):
        policy.act(frozen)
        if policy.phase != "transit":
            break
    assert policy.phase == "transfer", (
        f"stalled waypoint within 2 cm must advance, got {policy.phase}"
    )
    assert policy.retries == 0


def test_done_keeps_watching_and_repairs_a_piece_that_creeps_away():
    env = Env("panda")
    obs = env.reset([5], ["house"])[0]
    policy = Policy("panda", 5)
    policy.act(obs)
    solved = dict(obs)
    solved["pieces"] = np.array(policy.targets, dtype=float)
    solved["piece_velocities"] = np.zeros((7, 6))
    policy.index = 7
    policy.transition("start", solved)
    policy.act(solved)
    assert policy.phase == "done"
    crept = dict(solved)
    crept["pieces"] = solved["pieces"].copy()
    crept["pieces"][6, :2] += 0.05  # the parallelogram slid 5 cm after the hold began
    for _ in range(60):
        policy.act(crept)
    assert policy.phase == "approach" and policy.piece == 6 and policy.repairs == 1


def test_no_repair_starts_that_cannot_finish_before_the_horizon():
    env = Env("panda")
    obs = env.reset([4], ["cat"])[0]
    policy = Policy("panda", 4)
    policy.act(obs)
    solved = dict(obs)
    solved["pieces"] = np.array(policy.targets, dtype=float)
    solved["piece_velocities"] = np.zeros((7, 6))
    policy.index = 7
    policy.transition("start", solved)
    policy.act(solved)
    assert policy.phase == "done"
    crept = dict(solved)
    crept["pieces"] = solved["pieces"].copy()
    crept["pieces"][3, :2] += 0.05
    policy.elapsed = policy.horizon - 1500  # 30 s left: a repair would end in mid-air
    for _ in range(60):
        policy.act(crept)
    assert policy.phase == "done" and policy.repairs == 0


def test_done_returns_the_arm_home_and_follows_an_overridden_scene():
    from env import HOME

    env = Env("panda")
    obs = env.reset([3], ["house"], scenes=[{"goal_yaw": 1.0}])[0]
    policy = Policy("panda", 3)
    policy.act(obs)
    # The certificate follows the scene the environment actually built.
    assert policy.yaw == pytest.approx(1.0)
    solved = dict(obs)
    solved["pieces"] = np.array(policy.targets, dtype=float)
    solved["piece_velocities"] = np.zeros((7, 6))
    policy.index = 7
    policy.transition("start", solved)
    actions = [policy.act(solved) for _ in range(400)]
    assert policy.phase == "done"
    assert np.allclose(actions[-1][:7], HOME["panda"], atol=1e-6), "arm ends at home"
    steps = np.abs(np.diff(np.array(actions)[:, :7], axis=0)).max()
    assert steps < 0.03, "the return home glides"
