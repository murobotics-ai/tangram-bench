"""Tool-calling agent: interpolation and IK, rejections, budget, both wires, eval run."""

import io
import json

import mujoco
import numpy as np
import pytest

import adapters
import agent
from agent import GRASP_HEIGHT, AgentPolicy, downward, hand_yaw
from env import Env
from eval import main


def anthropic_reply(name, args, text="", stop="tool_use"):
    content = ([{"type": "text", "text": text}] if text else []) + (
        [{"type": "tool_use", "id": "toolu_1", "name": name, "input": args}] if name else []
    )
    return {
        "content": content,
        "stop_reason": stop,
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3},
    }


def openai_reply(name, args, text=""):
    output = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "x", "summary": []}]
    if text:
        output.append({"type": "message", "content": [{"type": "output_text", "text": text}]})
    if name:
        output.append(
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": name,
                "arguments": json.dumps(args),
            }
        )
    return {"output": output, "usage": {"input_tokens": 10, "output_tokens": 5}}


def serve(monkeypatch, replies):
    """Replace HTTPS by a queue of canned replies; returns the captured requests."""
    calls = []

    def post(request, timeout):
        calls.append(json.loads(request.data))
        if not replies:
            raise AssertionError("no canned reply left")
        return io.BytesIO(json.dumps(replies.pop(0)).encode())

    monkeypatch.setattr(agent, "urlopen", post)
    return calls


def make(monkeypatch, provider, **overrides):
    monkeypatch.setenv("TEST_TANGRAM_KEY", "test-only")
    config = {
        "type": "agent",
        "provider": provider,
        "model": "test-model",
        "api_key_env": "TEST_TANGRAM_KEY",
        "timeout_seconds": 5,
        "chunk_size": 25,
        "max_calls": 10,
        "effort": "low",
    }
    config.update(overrides)
    return adapters.policy_factory(config)("panda", 3)


def observe(env, index=0):
    obs = env.observe()[index]
    obs["images"] = env.images(index)
    return obs


def test_move_to_is_interpolated_solved_and_sliced(monkeypatch):
    env = Env(pixels=True)
    obs = observe(env.reset([3]) and env)
    target = {"x": 0.45, "y": -0.20, "z": 0.15, "yaw": 30.0, "gripper": 1.0, "note": "go"}
    calls = serve(monkeypatch, [anthropic_reply("move_to", target)])
    policy = make(monkeypatch, "anthropic")
    first = policy.act(obs)
    assert first.shape == (25, 8) and policy.usage["requests"] == 1
    assert policy.subtask == "go" and policy.step == 1 and policy.access == "pixels"
    # The system prompt carries the rig notes; the observation carries the cameras.
    assert calls[0]["system"].startswith("You are controlling a Franka Panda")
    blocks = calls[0]["messages"][0]["content"]
    assert [b["type"] for b in blocks].count("image") == 2
    assert "tool: x=" in blocks[0]["text"] and "Instruction:" in blocks[0]["text"]
    assert calls[0]["tools"][0]["name"] == "move_to" and "test-only" not in json.dumps(
        policy.audit()
    )
    # The rest of the motion comes out without another request, at 50 Hz and the
    # configured speed, and every joint target respects the actuator range.
    chunks = [first]
    while policy.queue:
        chunks.append(policy.act(obs))
    actions = np.vstack(chunks)
    assert policy.usage["requests"] == 1
    distance = np.linalg.norm(obs["tcp_pos"] - [0.45, -0.20, 0.15])
    assert abs(len(actions) - np.ceil(distance / 0.06 / 0.02)) <= 1
    env.validate_actions(actions)
    assert np.all(np.abs(np.diff(actions[:, :7], axis=0)) <= 0.04 + 1e-9)
    # The final joint target puts the tool point at the target pose.
    d = mujoco.MjData(env.model)
    d.qpos[:7] = actions[-1, :7]
    mujoco.mj_kinematics(env.model, d)
    tcp = env.model.site("tcp").id
    np.testing.assert_allclose(d.site_xpos[tcp], [0.45, -0.20, 0.15], atol=0.004)
    assert abs(np.rad2deg(hand_yaw(d.site_xmat[tcp].reshape(3, 3))) - 30) < 2
    np.testing.assert_allclose(d.site_xmat[tcp].reshape(3, 3), downward(np.deg2rad(30)), atol=0.05)
    # Executing the chunk in the simulator actually gets there.
    for row in actions:
        obs = env.step([row])[0]
    np.testing.assert_allclose(obs["tcp_pos"], [0.45, -0.20, 0.15], atol=0.01)
    env.close()


def test_gripper_change_takes_a_second_and_done_holds(monkeypatch):
    env = Env(pixels=True)
    obs = observe(env.reset([3]) and env)
    replies = [
        openai_reply(
            "move_by", {"dx": 0, "dy": 0, "dz": 0, "dyaw": 0, "gripper": 0, "note": "close"}
        ),
        openai_reply("done", {"summary": "s", "hindsight": "knob is 4 cm tall"}),
    ]
    calls = serve(monkeypatch, replies)
    policy = make(monkeypatch, "openai", chunk_size=100)
    chunk = policy.act(obs)
    assert len(chunk) == 50 and chunk[0, 7] < 1 and chunk[-1, 7] == 0
    assert calls[0]["tools"][0]["strict"] is True and calls[0]["store"] is False
    assert calls[0]["include"] == ["reasoning.encrypted_content"]
    for row in chunk:
        obs = env.step([row])[0]
    obs["images"] = env.images(0)
    held = policy.act(obs)
    assert held.shape == (8,) and policy.stopped["hindsight"] == "knob is 4 cm tall"
    # The second request echoes the reasoning item and the tool output back.
    items = calls[1]["input"]
    assert any(
        i.get("type") == "function_call_output" and i["output"].startswith("ok") for i in items
    )
    assert any(i.get("type") == "reasoning" for i in items)
    assert "last tool result: ok" in items[-1]["content"][0]["text"]
    # Holding: no more requests, same action forever.
    np.testing.assert_array_equal(policy.act(obs), held)
    assert policy.usage["requests"] == 2 and policy.audit()["stopped"]["tool"] == "done"
    env.close()


def test_rejections_nudges_and_budget(monkeypatch):
    env = Env(pixels=True)
    obs = observe(env.reset([3]) and env)
    far = {"x": 0.95, "y": 0.0, "z": 0.60, "yaw": 0.0, "gripper": 1.0, "note": "far"}
    calls = serve(monkeypatch, [anthropic_reply("move_to", far)] * 3)
    policy = make(monkeypatch, "anthropic")
    with pytest.raises(RuntimeError, match="rejected"):
        policy.act(obs)
    result = calls[1]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["content"].startswith("rejected")
    assert "x clamped from 0.950 to 0.800" not in result["content"]  # rejected before clamps report
    # No tool call at all: a nudge, then a policy error on the third miss.
    serve(monkeypatch, [anthropic_reply(None, None, text="thinking...", stop="end_turn")] * 3)
    policy = make(monkeypatch, "anthropic")
    with pytest.raises(RuntimeError, match="no tool call"):
        policy.act(obs)
    # Budget: the second act after exhausting max_calls is a forced give_up that holds.
    ok = {"x": 0.40, "y": -0.25, "z": 0.20, "yaw": 0.0, "gripper": 1.0, "note": "n"}
    serve(monkeypatch, [anthropic_reply("move_to", ok)])
    policy = make(monkeypatch, "anthropic", max_calls=1, chunk_size=1000)
    policy.act(obs)
    held = policy.act(obs)
    assert policy.stopped["forced"] and held.shape == (8,)
    assert policy.audit()["transcript"][-1]["forced_stop"]["reason"].startswith("tool call budget")
    env.close()


def test_image_horizon_and_state_mode(monkeypatch):
    env = Env(pixels=True)
    obs = observe(env.reset([3]) and env)
    move = {"dx": 0.0, "dy": 0.0, "dz": 0.02, "dyaw": 0.0, "gripper": 1.0, "note": "up"}
    calls = serve(monkeypatch, [anthropic_reply("move_by", move)] * 3)
    policy = make(monkeypatch, "anthropic", image_horizon=1, use_state=True, chunk_size=1000)
    for _ in range(3):
        policy.act(obs)
    assert policy.access == "state"
    text = calls[0]["messages"][0]["content"][0]["text"]
    assert "piece 'blue square'" in text and "goal outline vertices" in text
    images = [
        sum(b["type"] == "image" for b in m["content"])
        for m in calls[2]["messages"]
        if m["role"] == "user" and m["content"][0]["type"] == "text"
    ]
    assert images == [0, 0, 2]  # Only the newest observation keeps its cameras.
    assert "[image omitted]" in json.dumps(calls[2]["messages"][0])
    env.close()


def test_agent_system_runs_through_eval(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_TANGRAM_KEY", "test-only")
    grasp = {
        "x": 0.36,
        "y": -0.30,
        "z": GRASP_HEIGHT + 0.1,
        "yaw": 0.0,
        "gripper": 1.0,
        "note": "n",
    }
    serve(
        monkeypatch,
        [
            anthropic_reply("move_to", grasp),
            anthropic_reply("give_up", {"reason": "r", "hindsight": "h"}),
        ],
    )
    system = tmp_path / "agent.json"
    system.write_text(
        json.dumps(
            {
                "type": "agent",
                "provider": "anthropic",
                "model": "test-model",
                "api_key_env": "TEST_TANGRAM_KEY",
                "chunk_size": 25,
                "max_calls": 5,
                "timeout_seconds": 5,
            }
        )
    )
    out = tmp_path / "agent-run.json"
    assert (
        main(
            [
                "--system",
                str(system),
                "--obs",
                "pixels",
                "--max-chunk",
                "25",
                "--episodes",
                "1",
                "--steps",
                "400",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    row = result["episodes"][0]
    assert row["status"] == "completed" and row["policy_access"] == "pixels"
    assert row["policy_usage"]["requests"] == 2
    audit = json.loads((out.parent / row["policy_audit"]).read_text())
    assert audit["stopped"]["tool"] == "give_up" and audit["moves"] == 1
    assert "base64" not in json.dumps(audit["transcript"][0]["request"])
    trace = np.load(out.parent / f"{out.stem}.artifacts" / f"seed-{row['seed']}.npz")
    assert set(np.unique(trace["subtask"])) == {"n"} and trace["step"].max() == 1


def test_recipe_picks_carries_and_places_a_piece(monkeypatch):
    """The system prompt's recipe, followed literally, moves a piece 25 cm and sets
    it down flat within 15 mm. Guards the gripper command: fingers closed on the
    2 cm knob read 0.25 open, and interpolating from that reading let go of the
    piece at the start of every carry."""
    from tangram import THICKNESS

    env = Env(pixels=True)
    obs = observe(env.reset([3], ["house"]) and env)
    piece = obs["pieces"][0]
    yaw = float(np.rad2deg(2 * np.arctan2(piece[6], piece[3])))
    kx, ky = float(piece[0]), float(piece[1])
    tx, ty = 0.35, 0.0  # Between the packed square and the goal: clear of both.
    hold = {"gripper": 0.0, "note": "n"}
    moves = [
        ("move_to", {"x": kx, "y": ky, "z": 0.15, "yaw": yaw, "gripper": 1.0, "note": "n"}),
        ("move_to", {"x": kx, "y": ky, "z": GRASP_HEIGHT, "yaw": yaw, "gripper": 1.0, "note": "n"}),
        ("move_by", {"dx": 0, "dy": 0, "dz": 0, "dyaw": 0, **hold}),
        ("move_to", {"x": kx, "y": ky, "z": 0.15, "yaw": yaw, **hold}),
        ("move_to", {"x": tx, "y": ty, "z": 0.15, "yaw": yaw, **hold}),
        ("move_to", {"x": tx, "y": ty, "z": agent.RELEASE_HEIGHT, "yaw": yaw, **hold}),
        ("move_by", {"dx": 0, "dy": 0, "dz": 0, "dyaw": 0, "gripper": 1.0, "note": "n"}),
        ("move_to", {"x": tx, "y": ty, "z": 0.15, "yaw": yaw, "gripper": 1.0, "note": "n"}),
        ("done", {"summary": "s", "hindsight": "h"}),
    ]
    serve(monkeypatch, [anthropic_reply(name, args) for name, args in moves])
    policy = make(monkeypatch, "anthropic", max_calls=20)
    ticks, lifted = 0, 0.0
    while policy.stopped is None and ticks < 3000:
        action = policy.act(obs)
        for row in action if action.ndim == 2 else action[None]:
            obs = env.step([row])[0]
            ticks += 1
            lifted = max(lifted, float(obs["pieces"][0][2]))
        obs["images"] = env.images(0)
    final = obs["pieces"][0]
    assert lifted > 0.10, "the piece never left the table"
    assert np.hypot(final[0] - tx, final[1] - ty) < 0.015
    assert abs(final[2] - THICKNESS / 2) < 0.002 and 1 - 2 * (final[4] ** 2 + final[5] ** 2) > 0.999
    # Every carry command kept the pinch: the gripper column never rose above 0 while held.
    carried = [
        r for r in policy.records if r.get("tool_result", {}).get("result", "").startswith("ok")
    ]
    assert "gripper 0.00 -> 0.00" in carried[3]["tool_result"]["result"]
    env.close()


def test_agent_rejects_bad_configuration(monkeypatch):
    monkeypatch.setenv("TEST_TANGRAM_KEY", "k")
    base = {"type": "agent", "model": "m", "api_key_env": "TEST_TANGRAM_KEY"}
    with pytest.raises(ValueError, match="provider"):
        AgentPolicy("panda", 0, {**base, "provider": "cohere"})
    with pytest.raises(ValueError, match="Panda"):
        AgentPolicy("piper", 0, {**base, "provider": "openai"})
    with pytest.raises(ValueError, match="configuration"):
        AgentPolicy("panda", 0, {**base, "provider": "openai", "speed": 2.0})
