"""Corpus, prompt, provider boundaries and stored episode inspection."""

import io
import json

import mujoco
import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

import adapters
from env import Env
from eval import main
from replay import Replay
from shapes import PROMPT, SUITES, TARGETS, canonical, solution, validate
from tangram import goal, polygons, score
from tools.history import collect
from view import PROMPT_HEIGHT, goal_triangles, layout


@pytest.mark.parametrize("target", TARGETS)
def test_every_shape_has_seven_unreflected_nonoverlapping_pieces(target):
    assert validate(target)["iou_ceiling"] >= 0.95
    for seed in [0, 100000, 200000]:
        poses = solution(target, seed)
        assert score(poses, np.zeros((7, 6)), goal(seed, target))["success"]
        footprints = polygons(poses)
        assert sum(p.area for p in footprints) - unary_union(footprints).area < 1e-10
    outline, _, _ = canonical(target)
    triangles = goal_triangles(tuple(map(tuple, outline)))
    assert (
        unary_union([Polygon(t) for t in triangles]).symmetric_difference(Polygon(outline)).area
        < 1e-10
    )


def test_shape_splits_are_disjoint_and_not_rotated_copies():
    assert set(SUITES["train"]).isdisjoint(SUITES["dev"])
    assert set(SUITES["test"]).isdisjoint(SUITES["train"] + SUITES["dev"])
    # Perimeter is invariant under rigid motion and distinguishes this pilot corpus.
    perimeters = [round(Polygon(canonical(t)[0]).length, 8) for t in TARGETS]
    assert len(set(perimeters)) == len(TARGETS)


@pytest.mark.parametrize("robot", ["panda", "piper"])
@pytest.mark.parametrize("target", TARGETS)
def test_reference_poses_settle_and_source_stays_clear(robot, target):
    env = Env(robot)
    obs = env.reset([100000], [target])[0]
    assert obs["prompt"] == PROMPT and obs["target"] == target
    assert all(p.distance(Polygon(obs["goal"])) > 0.01 for p in polygons(obs["pieces"]))
    d = env.data[0]
    for address, pose in zip(env.qadr, solution(target, 100000)):
        d.qpos[address : address + 7] = pose
    mujoco.mj_forward(env.model, d)
    mujoco.mj_step(env.model, d, nstep=100)
    obs = env.observe()[0]
    assert score(obs["pieces"], obs["piece_velocities"], obs["goal"])["success"]


def test_prompt_is_optional_for_policy_but_always_recorded(tmp_path):
    out = tmp_path / "prompt.json"
    prompt = "Assemble the tangram to match the silhouette."
    assert main(["--episodes", "1", "--steps", "3", "--prompt", prompt, "--out", str(out)]) == 0
    replay = Replay(out)
    assert replay.prompt == prompt and replay.target == "house"
    replay.seek(3)
    np.testing.assert_array_equal(replay.data[0].qpos[:9], replay.trace["qpos"][3])
    assert replay.steps == 3
    rows = collect(tmp_path)
    assert len(rows) == 1 and rows[0]["verification"] == "verified"
    assert rows[0]["prompt"] == prompt
    main_rect, _ = layout(1440, 960)
    assert main_rect.height == 960 - PROMPT_HEIGHT


@pytest.mark.parametrize("kind", ["http", "openai", "anthropic"])
def test_provider_specific_payloads_and_response_parsing(kind, monkeypatch):
    monkeypatch.setenv("TEST_TANGRAM_KEY", "test-only")
    calls = []
    action = [0, -0.45, 0, -2.2, 0, 1.8, 0.7854, 1]
    content = json.dumps({"actions": [action]})
    response = (
        {"actions": [action]}
        if kind == "http"
        else (
            {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": content}]}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
            if kind == "openai"
            else {
                "content": [{"type": "text", "text": content}],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        )
    )

    def post(request, timeout):
        calls.append(json.loads(request.data))
        assert timeout == 5
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(adapters, "urlopen", post)
    config = {
        "type": kind,
        "model": "test-model",
        "url": "https://example.invalid/infer",
        "api_key_env": "TEST_TANGRAM_KEY",
        "timeout_seconds": 5,
        "use_prompt": True,
    }
    policy = adapters.policy_factory(config)("panda", 3)
    obs = Env().reset([3])[0]
    np.testing.assert_array_equal(policy.act(obs), [action])
    assert PROMPT in json.dumps(calls[0])
    assert policy.audit()["usage"]["requests"] == 1
    assert "test-only" not in json.dumps(policy.audit())
    if kind == "http":
        assert "observation" in calls[0]
    elif kind == "openai":
        assert calls[0]["store"] is False and "input" in calls[0]
    else:
        assert calls[0]["messages"][0]["role"] == "user"


def test_checkpoint_config_passes_weights_to_custom_local_policy(tmp_path):
    module = tmp_path / "local.py"
    module.write_text(
        "import numpy as np\nclass Policy:\n    def __init__(self,robot,seed,checkpoint):\n        self.target=np.load(checkpoint)\n    def act(self,obs):\n        return self.target.copy()\n"
    )
    weights = tmp_path / "checkpoint.npy"
    np.save(weights, np.arange(8, dtype=float))
    config = tmp_path / "system.json"
    config.write_text(
        json.dumps({"type": "local", "module": "local.py", "checkpoint": "checkpoint.npy"})
    )
    resolved, files = adapters.system_config(config)
    assert set(files) == {config, module, weights}
    p = adapters.policy_factory(resolved)("panda", 0)
    np.testing.assert_array_equal(p.act({}), np.arange(8))


def test_http_adapter_can_omit_prompt(monkeypatch):
    sent = []

    def post(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'{"actions": [[0,0,0,0,0,0,0,1]]}')

    monkeypatch.setattr(adapters, "urlopen", post)
    p = adapters.RemotePolicy(
        "panda", 0, {"type": "http", "url": "https://example.invalid", "use_prompt": False}
    )
    p.act(Env().reset([0])[0])
    assert "prompt" not in sent[0]["observation"]
    assert p.audit()["usage"]["input_tokens"] is None
    assert p.audit()["usage"]["output_tokens"] is None


def test_history_separates_seeds_and_exposes_checkpoint_identity(tmp_path):
    for index, seed in enumerate([100000, 100001]):
        result = {
            "started_at": f"2026-09-08T12:00:0{index}+00:00",
            "finished_at": "2026-09-08T12:01:00+00:00",
            "episodes": [],
            "seeds": [seed],
            "policy": "/system.json",
            "policy_sha256": "config-hash",
            "policy_artifacts": {"/weights.npy": "checkpoint-hash"},
            "policy_metadata": {"system": {"checkpoint": "/weights.npy"}},
        }
        (tmp_path / f"run-{index}.json").write_text(json.dumps(result))
    rows = collect(tmp_path)
    assert rows[0]["cohort"] != rows[1]["cohort"]
    assert rows[0]["checkpoint_sha256"] == "checkpoint-hash"
    assert rows[0]["policy_sha256"] == "config-hash"
    assert all(row["verification"] != "verified" for row in rows)
