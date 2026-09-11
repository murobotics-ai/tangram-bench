"""Corpus, prompt, provider boundaries and stored episode inspection."""

import io
import json
import os

import mujoco
import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

import adapters
from env import Env, goal_triangles
from eval import main
from replay import Replay
from shapes import SUITES, TARGETS, canonical, prompt, solution, validate
from tangram import goal, polygons, score
from tools.history import collect
from view import PROMPT_HEIGHT, layout


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
    # Dev reuses the training figures on unseen seeds; test is the unseen figure.
    assert set(SUITES["dev"]) == set(SUITES["train"]) and len(SUITES["train"]) == 3
    assert set(SUITES["test"]).isdisjoint(SUITES["train"])

    # Sorted edge lengths are invariant under rigid motion and distinguish the corpus.
    def edges(target):
        outline = canonical(target)[0]
        return tuple(sorted(np.round(np.linalg.norm(np.roll(outline, -1, 0) - outline, axis=1), 6)))

    assert len({edges(t) for t in TARGETS}) == len(TARGETS)


@pytest.mark.parametrize("robot", ["panda", "piper"])
@pytest.mark.parametrize("target", TARGETS)
def test_reference_poses_settle_and_source_stays_clear(robot, target):
    env = Env(robot)
    obs = env.reset([100000], [target])[0]
    assert obs["prompt"] == prompt(target) and obs["target"] == target
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
    custom = "Assemble the tangram to match the silhouette."
    assert main(["--episodes", "1", "--steps", "3", "--prompt", custom, "--out", str(out)]) == 0
    replay = Replay(out)
    assert replay.prompt == custom and replay.target == SUITES["dev"][0]
    assert replay.label() == "" and replay.frame_seconds == 0.02
    replay.seek(3)
    np.testing.assert_array_equal(replay.data[0].qpos[:9], replay.trace["qpos"][3])
    assert replay.steps == 3
    rows = collect(tmp_path)
    assert len(rows) == 1 and rows[0]["verification"] == "verified"
    assert rows[0]["prompt"] == custom
    default = tmp_path / "default.json"
    assert main(["--episodes", "1", "--steps", "2", "--target", "cat", "--out", str(default)]) == 0
    assert (
        Replay(default).prompt == prompt("cat") == "Solve the tangram puzzle to assemble the cat."
    )
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
    env = Env(pixels=True)
    obs = env.reset([3])[0]
    obs["images"] = env.images(0)
    np.testing.assert_array_equal(policy.act(obs), [action])
    assert prompt("square") in json.dumps(calls[0])
    assert policy.audit()["usage"]["requests"] == 1
    assert "test-only" not in json.dumps(policy.audit())
    if kind == "http":
        assert "observation" in calls[0] and set(calls[0]["images"]) == {"top", "wrist"}
    elif kind == "openai":
        assert calls[0]["store"] is False and calls[0]["text"]["format"]["strict"] is True
        parts = calls[0]["input"][0]["content"]
        assert [p["type"] for p in parts].count("input_image") == 2
        assert parts[1]["image_url"].startswith("data:image/png;base64,")
        assert parts[-1]["type"] == "input_text" and "Observation" in parts[-1]["text"]
    else:
        blocks = calls[0]["messages"][0]["content"]
        assert [b["type"] for b in blocks].count("image") == 2
        assert blocks[1]["source"]["media_type"] == "image/png"
        assert calls[0]["output_config"]["format"]["type"] == "json_schema"
        assert calls[0]["system"].startswith("You control a panda")
    env.close()


def test_png_encoder_round_trips_and_dotenv_never_overrides(tmp_path, monkeypatch):
    from PIL import Image

    image = np.random.default_rng(0).integers(0, 255, (6, 5, 3), dtype=np.uint8)
    decoded = np.asarray(Image.open(io.BytesIO(adapters.png(image))).convert("RGB"))
    np.testing.assert_array_equal(decoded, image)
    dotenv = tmp_path / ".env"
    dotenv.write_text("# keys\nTEST_TANGRAM_A='from-file'\nTEST_TANGRAM_B=x\n\nbroken line\n")
    monkeypatch.setenv("TEST_TANGRAM_B", "from-shell")
    monkeypatch.delenv("TEST_TANGRAM_A", raising=False)
    assert adapters.load_env(dotenv) == {"TEST_TANGRAM_A": "from-file"}
    assert (
        os.environ["TEST_TANGRAM_A"] == "from-file" and os.environ["TEST_TANGRAM_B"] == "from-shell"
    )
    assert adapters.load_env(tmp_path / "missing") == {}


@pytest.mark.parametrize(
    "kind,reply,match",
    [
        (
            "openai",
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "incomplete",
        ),
        (
            "openai",
            {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]},
            "refusal",
        ),
        ("anthropic", {"stop_reason": "max_tokens", "content": []}, "truncated"),
        (
            "anthropic",
            {"stop_reason": "refusal", "content": [], "stop_details": {"category": "x"}},
            "refusal",
        ),
    ],
)
def test_provider_truncation_and_refusals_are_policy_errors(kind, reply, match, monkeypatch):
    monkeypatch.setenv("TEST_TANGRAM_KEY", "test-only")
    monkeypatch.setattr(
        adapters, "urlopen", lambda r, timeout: io.BytesIO(json.dumps(reply).encode())
    )
    policy = adapters.RemotePolicy(
        "panda", 0, {"type": kind, "model": "m", "api_key_env": "TEST_TANGRAM_KEY", "retries": 0}
    )
    with pytest.raises(RuntimeError, match=match):
        policy.act(Env().reset([0])[0])


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


def test_scene_design_is_a_documented_grid_indexed_by_seed():
    import numpy as np

    from benchmark import SPLITS
    from tangram import GOAL_OFFSETS, SOURCE_OFFSETS, SPLIT_SIZE, describe_layout, layout

    assert SPLITS == {"train": 0, "dev": SPLIT_SIZE, "test": 2 * SPLIT_SIZE}
    train = [describe_layout(layout(n, "house")) for n in range(60)]
    assert {s["goal_yaw_deg"] for s in train} == {30.0 * k for k in range(12)}
    assert {s["source_yaw_deg"] for s in train} == {45.0 * k for k in range(8)}
    assert len({tuple(s["goal_center_mm"]) for s in train}) == len(GOAL_OFFSETS)
    assert len({tuple(s["source_center_mm"]) for s in train}) == len(SOURCE_OFFSETS)
    # Dev rotations fall exactly between the training ones; the test split reuses the grid.
    dev = {describe_layout(layout(SPLITS["dev"] + n, "cat"))["goal_yaw_deg"] for n in range(60)}
    assert dev == {15.0 + 30.0 * k for k in range(12)}
    assert describe_layout(layout(SPLITS["test"] + 3, "cat"))["goal_yaw_deg"] == 90.0
    assert layout(7, "square")["goal_yaw"] == layout(7, "square")["goal_yaw"]  # deterministic
    assert np.allclose(
        layout(7, "square")["goal_center"] - layout(7, "house")["goal_center"], [0, -0.03]
    )


def test_scene_override_is_applied_and_recorded(tmp_path):
    import json

    from env import Env
    from tools.collect import main as collect

    env = Env("panda")
    env.reset([4], ["house"], scenes=[{"goal_yaw": 0.5, "source_yaw": None}])
    assert env.scenes[0]["goal_yaw"] == 0.5 and env.scenes[0]["source_yaw"] != 0.5
    collect(
        [
            "--target",
            "house",
            "--episodes",
            "1",
            "--steps",
            "10",
            "--policy",
            "policy.py",
            "--keep-failures",
            "--goal-yaw",
            "45",
            "--out",
            str(tmp_path),
            "--round",
            "r",
        ]
    )
    row = json.loads((tmp_path / "r" / "index.jsonl").read_text().splitlines()[0])
    assert row["goal_yaw_deg"] == 45.0 and "source_yaw_deg" in row and "goal_center_mm" in row


def test_every_designed_scene_keeps_the_pieces_inside_the_workspace():
    from benchmark import SPLITS
    from tangram import WORKSPACE, layout, piece_radii

    for target in ("square", "rectangle", "house", "cat"):
        for base in (SPLITS["train"], SPLITS["dev"], SPLITS["test"]):
            for n in range(60):
                radii = piece_radii(layout(base + n, target), target)
                assert radii.min() >= WORKSPACE[0] and radii.max() <= WORKSPACE[1], (
                    target,
                    base + n,
                )
