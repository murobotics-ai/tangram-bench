"""Pixel observations, the pixel evaluation track, demonstration recording and export."""

import importlib.util
import json

import numpy as np
import pytest

from benchmark import COMPARISON_FIELDS, digest
from env import CAMERAS, IMAGE_SIZE, Env
from eval import main
from tools.collect import main as collect
from tools.results import verify

HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None


@pytest.fixture(autouse=True)
def rendering():
    """Skip rendering tests where no OpenGL context can be created (bare CI)."""
    try:
        env = Env("panda", pixels=True)
        env.reset([0])
        env.images(0)
        env.close()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"offscreen rendering unavailable: {exc}")


def test_pixels_are_rendered_on_demand_and_carry_the_goal():
    env = Env("panda", pixels=True)
    env.reset([100000], ["house"])
    images = env.images(0)
    assert set(images) == set(CAMERAS)
    for image in images.values():
        assert image.shape == (*IMAGE_SIZE, 3) and image.dtype == np.uint8
    again = Env("panda", pixels=True)
    again.reset([100000], ["house"])
    np.testing.assert_array_equal(again.images(0)["top"], images["top"])
    again.reset([100000], ["cat"])
    assert not np.array_equal(again.images(0)["top"], images["top"]), "goal not visible"
    env.close(), again.close()
    with pytest.raises(RuntimeError, match="pixels=True"):
        Env("panda").images(0)


def test_pixel_track_gives_images_only_to_inference_and_is_a_distinct_contract(tmp_path):
    candidate = tmp_path / "pixel_policy.py"
    candidate.write_text(
        "import numpy as np\n"
        "class Policy:\n"
        "    access = 'pixels'\n"
        "    def __init__(self, robot, seed): self.narm = 7\n"
        "    def act(self, obs):\n"
        "        assert obs['images']['top'].shape == (240, 320, 3), obs.keys()\n"
        "        return np.tile(np.r_[obs['qpos'][:7], 1.0], (2, 1))\n"
    )
    out = tmp_path / "pixels.json"
    args = ["--policy", str(candidate), "--episodes", "1", "--steps", "4", "--max-chunk", "2"]
    assert main([*args, "--obs", "pixels", "--out", str(out)]) == 0
    run = verify(out)
    assert run["observation"] == "pixels" and run["cameras"] == list(CAMERAS)
    assert run["episodes"][0]["inference_calls"] == 2
    assert "observation" in COMPARISON_FIELDS
    state = tmp_path / "state.json"
    assert main([*args, "--out", str(state)]) == 0
    assert json.loads(state.read_text())["episodes"][0]["status"] == "policy_error"


def test_collect_records_frames_at_fps_with_interval_end_actions(tmp_path):
    collect(
        [
            "--target",
            "square",
            "--episodes",
            "1",
            "--steps",
            "12",
            "--fps",
            "10",
            "--policy",
            "policy.py",
            "--keep-failures",
            "--out",
            str(tmp_path),
        ]
    )
    with np.load(tmp_path / "episode-0.npz", allow_pickle=False) as data:
        assert data["images_top"].shape == (3, *IMAGE_SIZE, 3)
        assert data["state"].shape == (3, 9) and data["action"].shape == (3, 8)
        np.testing.assert_allclose(data["time"], [0.0, 0.1, 0.2], atol=1e-6)
        assert not data["success"] and str(data["target"]) == "square"
        assert data["action"][0, -1] == 1.0 and data["fps"] == 10
        assert data["subtask"].shape == (3,) and str(data["prompt"]).endswith("the square.")
        assert data["step"].tolist() == [0, 0, 0] and data["plan"].shape == (0,)
    from replay import Demo

    demo = Demo(tmp_path / "episode-0.npz")
    demo.seek(2)
    assert demo.steps == 2 and demo.frame_seconds == 0.1 and demo.label() == ""
    assert demo.text().startswith("DEMO | seed 0 | not solved")
    rows = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert rows == [{**rows[0], "seed": 0, "frames": 3, "success": False}]
    with pytest.raises(SystemExit):
        collect(["--target", "square", "--fps", "7", "--out", str(tmp_path)])


def test_directory_digest_tracks_every_file(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    before = digest(tmp_path)
    (tmp_path / "b.txt").write_text("b")
    assert digest(tmp_path) != before
    assert digest(tmp_path) == digest(tmp_path)


@pytest.mark.skipif(not HAS_LEROBOT, reason="uv sync --extra lerobot")
def test_export_and_lerobot_policy_round_trip(tmp_path):
    from tools.export import export

    collect(
        [
            "--target",
            "house",
            "--episodes",
            "1",
            "--steps",
            "10",
            "--fps",
            "10",
            "--policy",
            "policy.py",
            "--keep-failures",
            "--out",
            str(tmp_path / "raw"),
        ]
    )
    rows = export([tmp_path / "raw"], tmp_path / "lerobot", "test/tangram", include_failures=True)
    info = json.loads((tmp_path / "lerobot" / "meta" / "info.json").read_text())
    assert rows[0]["frames"] == 2 and info["fps"] == 10
    assert rows[0]["subtasks"] == [{"frame": 0, "text": ""}] and rows[0]["plan"] == []
    assert rows[0]["prompt"] == "Solve the tangram puzzle to assemble the house."
    assert set(info["features"]) >= {"language_persistent", "language_events"}
    assert set(info["features"]) >= {"observation.images.top", "observation.state", "action"}

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    from examples.lerobot_policy import Policy

    dataset = LeRobotDataset("test/tangram", root=tmp_path / "lerobot")
    assert dataset.meta.has_language_columns and dataset[0]["language_persistent"] == []
    config = ACTConfig(
        input_features={
            f"observation.images.{name}": PolicyFeature(FeatureType.VISUAL, (3, *IMAGE_SIZE))
            for name in CAMERAS
        }
        | {"observation.state": PolicyFeature(FeatureType.STATE, (9,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
        chunk_size=10,
        n_action_steps=10,
        dim_model=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=2,
        dim_feedforward=64,
        pretrained_backbone_weights=None,
    )
    config.device = "cpu"
    checkpoint = tmp_path / "act"
    ACTPolicy(config).save_pretrained(checkpoint)
    for processor in make_pre_post_processors(config, dataset_stats=dataset.meta.stats):
        processor.save_pretrained(checkpoint)
    env = Env("panda", pixels=True)
    obs = env.reset([100000], ["house"])[0]
    obs["images"] = env.images(0)
    policy = Policy("panda", 0, checkpoint=checkpoint, fps=10, n_action_steps=4, device="cpu")
    actions = env.validate_actions(policy.act(obs))
    assert actions.shape == (20, 8)
    np.testing.assert_array_equal(actions[0], actions[4])
    env.close()


def test_language_rows_follow_lerobot_annotate_conventions():
    from tools.export import language_rows

    data = {
        "fps": 10,
        "subtask": np.array(["reach", "reach", "carry", "carry", "reach", "carry"]),
        "step": np.array([1, 1, 1, 1, 2, 2]),
        "plan": np.array(["Place A.", "Place B."]),
    }
    rows = language_rows(data)
    assert [(r["style"], r["timestamp"]) for r in rows] == [
        ("plan", 0.0),
        ("subtask", 0.0),
        ("subtask", 0.2),
        ("plan", 0.4),
        ("subtask", 0.4),
        ("subtask", 0.5),
    ]
    assert rows[0]["content"] == "1. Place A.\n2. Place B." and rows[3]["content"] == "1. Place B."
    assert all(r["role"] == "assistant" and r["camera"] is None for r in rows)


def test_train_command_uses_repo_paths_and_passes_extra_flags(tmp_path, monkeypatch, capsys):
    from tools import train

    argv = train.command("smolvla", steps=500, extra=["--batch_size=2"])
    assert argv[0] == "lerobot-train" and "--policy.path=lerobot/smolvla_base" in argv
    assert "--policy.scheduler_decay_steps=500" in argv and "--steps=500" in argv
    assert "--output_dir=outputs/train/smolvla" in argv and argv[-1] == "--batch_size=2"
    assert "--policy.scheduler_decay_steps=500" not in train.command("act", steps=500)
    with pytest.raises(ValueError):
        train.command("pi05")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert train.main(["--policy", "act", "--dry-run", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(f"HF_HUB_CACHE={train.CHECKPOINTS} lerobot-train --policy.type=act")
    assert train.CHECKPOINTS == train.ROOT / "checkpoints"
