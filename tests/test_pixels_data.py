"""Pixel observations, the pixel evaluation track, demonstration recording and export."""

import importlib.util
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
    difference = np.abs(again.images(0)["top"].astype(int) - images["top"].astype(int))
    assert difference.max() <= 1, "renderers may differ by rasterization noise only"
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
            "--round",
            "r",
        ]
    )
    with np.load(tmp_path / "r" / "episode-0.npz", allow_pickle=False) as data:
        assert data["images_top"].shape == (3, *IMAGE_SIZE, 3)
        assert data["state"].shape == (3, 9) and data["action"].shape == (3, 8)
        np.testing.assert_allclose(data["time"], [0.0, 0.1, 0.2], atol=1e-6)
        assert not data["success"] and str(data["target"]) == "square"
        assert data["action"][0, -1] == 1.0 and data["fps"] == 10
        assert data["subtask"].shape == (3,) and str(data["prompt"]).endswith("the square.")
        assert data["step"].tolist() == [0, 0, 0] and data["plan"].shape == (0,)
    from replay import Demo

    demo = Demo(tmp_path / "r" / "episode-0.npz")
    demo.seek(2)
    assert demo.steps == 2 and demo.frame_seconds == 0.1 and demo.label() == ""
    assert demo.text().startswith("DEMO | seed 0 | not solved")
    rows = [json.loads(line) for line in (tmp_path / "r" / "index.jsonl").read_text().splitlines()]
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
def test_export_and_lerobot_policy_round_trip(tmp_path, monkeypatch):
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
    from tools import export as module

    text = module.card(rows, "panda", 10, "prompt")
    assert "| Episodes | 1 (0 solved) |" in text and "| house | 1 | 0 | " in text
    assert module.GITHUB in text and "| Frames | 2 at 10 fps" in text
    items, robot = module.scan([tmp_path / "raw"], include_failures=True)
    assert robot == "panda" and module.dataset_name(items, robot) == "tangram-house-panda-1ep"
    assert module.dataset_name(items, robot, "subtask") == "tangram-house-panda-subtask-1ep"
    assert module.scan([tmp_path / "raw"])[0] == []  # the episode failed
    calls = []
    monkeypatch.setattr(module, "push", lambda *a: calls.append(a) or "https://x")
    monkeypatch.chdir(tmp_path)
    module.main(
        [str(tmp_path / "raw"), "--include-failures", "--push", "--private", "--namespace", "org"]
    )
    out = tmp_path / "data" / "lerobot" / "tangram-house-panda-1ep"
    assert calls[0][0].resolve() == out and calls[0][1:] == (
        "org/tangram-house-panda-1ep",
        calls[0][2],
        "prompt",
        True,
    )
    assert (out / "episodes.jsonl").exists()
    info = json.loads((tmp_path / "lerobot" / "meta" / "info.json").read_text())
    assert rows[0]["frames"] == 2 and info["fps"] == 10
    assert rows[0]["subtasks"] == [{"frame": 0, "text": ""}] and rows[0]["plan"] == []
    assert rows[0]["prompt"] == "Solve the tangram puzzle to assemble the house."
    assert set(info["features"]) >= {"subtask_index", "plan_index", "task_index"}
    assert not {"language_persistent", "language_events"} & set(info["features"])
    assert set(info["features"]) >= {"observation.images.top", "observation.state", "action"}
    assert info["features"]["subtask_index"] == {"dtype": "int64", "shape": [1], "names": None}
    assert list(pd.read_parquet(tmp_path / "lerobot" / "meta" / "subtasks.parquet").index) == []
    assert list(pd.read_parquet(tmp_path / "lerobot" / "meta" / "plans.parquet").index) == []

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    from examples.lerobot_policy import Policy

    dataset = LeRobotDataset("test/tangram", root=tmp_path / "lerobot")
    assert not dataset.meta.has_language_columns
    assert dataset[0]["subtask_index"].item() == -1 and dataset[0]["plan_index"].item() == -1
    assert "subtask_index" in dataset.meta.stats and dataset.meta.stats["subtask_index"]["max"] == [
        -1
    ]
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


def test_relabel_maps_legacy_motion_sentences_to_plan_lines():
    from tools.reindex import relabel

    row = {
        "plan": [
            "Place the blue square at the top of the house.",
            "Place the red medium triangle at the left of the house.",
        ],
        "subtasks": [
            {"frame": 0, "text": "Reach for the blue square and grasp its knob from above."},
            {"frame": 5, "text": "Lift the blue square off the table."},
            {"frame": 9, "text": "Carry the blue square to the top of the house and align it."},
            {
                "frame": 20,
                "text": "The blue square slipped; let go, back away and pick it up again.",
            },
            {"frame": 30, "text": "Lower the blue square into place at the top of the house."},
            {"frame": 40, "text": "Release the blue square and back away."},
            {
                "frame": 50,
                "text": "Reach for the red medium triangle and grasp its knob from above.",
            },
            {"frame": 90, "text": "All seven pieces are placed; hold still."},
        ],
    }
    out = relabel(row)
    assert out["plan"] == [
        "Place the blue square at the top of the outline.",
        "Place the red medium triangle at the left of the outline.",
    ]
    assert out["subtasks"] == [
        {"frame": 0, "text": "Place the blue square at the top of the outline."},
        {"frame": 20, "text": "The blue square slipped; let go, back away and pick it up again."},
        {"frame": 30, "text": "Place the blue square at the top of the outline."},
        {"frame": 50, "text": "Place the red medium triangle at the left of the outline."},
        {"frame": 90, "text": "All seven pieces are placed; hold still."},
    ]


def test_vocabulary_and_plan_text_follow_the_tasks_parquet_layout(tmp_path):
    from tools.export import NONE, Vocabulary, plan_text

    words = Vocabulary("subtask")
    assert [words(t) for t in ["reach", "reach", "carry", "", "reach"]] == [0, 0, 1, NONE, 0]
    words.write(tmp_path)
    frame = pd.read_parquet(tmp_path / "meta" / "subtasks.parquet")
    assert frame.index.name == "subtask" and list(frame.index) == ["reach", "carry"]
    assert frame["subtask_index"].tolist() == [0, 1]
    assert plan_text(["Place A.", "Place B."]) == "1. Place A.\n2. Place B." and plan_text([]) == ""


@pytest.mark.skipif(not HAS_LEROBOT, reason="uv sync --extra lerobot")
def test_reindex_rebuilds_the_index_format_from_the_sidecar(tmp_path):
    from tools.export import export
    from tools.reindex import annotations, reindex

    collect(
        [
            "--target",
            "house",
            "--episodes",
            "2",
            "--steps",
            "10",
            "--fps",
            "10",
            "--policy",
            "examples/oracle.py",
            "--keep-failures",
            "--out",
            str(tmp_path / "raw"),
        ]
    )
    rows = export([tmp_path / "raw"], tmp_path / "lerobot", "test/tangram", include_failures=True)
    assert rows[0]["plan"] and rows[0]["subtasks"][0]["text"] == rows[0]["plan"][0]
    root = tmp_path / "lerobot"
    before = {p.relative_to(root): pq.read_table(p) for p in root.rglob("*.parquet")}
    plan, subtasks = annotations(root)[0]
    assert plan.startswith("1. ") and len(subtasks) == rows[0]["frames"]
    assert reindex(root) == {"subtask": 2, "plan": 2}  # first plan line per scene
    for p in root.rglob("*.parquet"):
        assert pq.read_table(p).equals(before[p.relative_to(root)]), p
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("test/tangram", root=root)
    sample = dataset[0]
    assert sample["subtask_index"].item() == 0 and sample["plan_index"].item() == 0
    assert pd.read_parquet(root / "meta" / "subtasks.parquet").index[0] == subtasks[0]


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


def test_collect_workers_split_seeds_and_match_the_sequential_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")  # Worker processes render headless.
    common = ["--target", "house", "--episodes", "3", "--steps", "10", "--fps", "10"]
    common += ["--policy", "policy.py", "--keep-failures"]
    collect([*common, "--workers", "3", "--out", str(tmp_path / "par"), "--round", "r"])
    collect([*common, "--out", str(tmp_path / "seq"), "--round", "r"])
    rows = [
        json.loads(line)
        for line in (tmp_path / "par" / "r" / "index.jsonl").read_text().splitlines()
    ]
    assert sorted(r["seed"] for r in rows) == [0, 1, 2] and all(r["frames"] == 2 for r in rows)
    for seed in range(3):
        with (
            np.load(tmp_path / "par" / "r" / f"episode-{seed}.npz") as a,
            np.load(tmp_path / "seq" / "r" / f"episode-{seed}.npz") as b,
        ):
            for key in ("state", "action", "pieces", "time", "goal", "subtask", "prompt"):
                assert np.array_equal(a[key], b[key]), key
            assert a["images_top"].shape == b["images_top"].shape == (2, *IMAGE_SIZE, 3)


def test_env_renders_the_context_camera_for_monitoring_only():
    env = Env("panda", pixels=True)
    env.reset([0], ["house"])
    assert env.render("context").shape == (*IMAGE_SIZE, 3)
    assert "context" not in env.images(0)  # Policies never receive the third-person view.
    env.close()


def test_fleet_model_places_each_world_on_its_own_table():
    import mujoco

    from env import fleet_model, make_model

    single = make_model("panda")
    model, offsets, index, shift = fleet_model("panda", 6, 3)
    assert model.nq == 6 * single.nq and len(np.unique(index)) == index.size
    assert offsets.shape == (6, 2) and shift.sum() == 14  # x and y of seven free joints
    env = Env("panda")
    env.reset([0], ["house"])
    data = mujoco.MjData(model)
    for i in range(6):
        data.qpos[index[i]] = env.data[0].qpos + shift @ offsets[i]
    mujoco.mj_forward(model, data)
    for i in range(6):
        piece = data.body(f"w{i}_piece3").xpos[:2] - offsets[i]
        np.testing.assert_allclose(piece, env.data[0].body("piece3").xpos[:2], atol=1e-9)
        arm = data.body(f"w{i}_hand").xpos - [*offsets[i], 0]
        np.testing.assert_allclose(arm, env.data[0].body("hand").xpos, atol=1e-9)


def test_recorder_preallocates_for_the_horizon_and_refuses_overflow():
    from tools.collect import Recorder

    env = Env("panda", pixels=True)
    obs = env.reset([0], ["square"])[0]
    recorder = Recorder(env, 0, 10, obs["prompt"], steps=20)  # capacity 20 // 5 + 2 = 6 frames
    action = np.r_[obs["qpos"][:7], 1.0]
    assert recorder.images["top"].shape == (6, *IMAGE_SIZE, 3) and recorder.count == 0
    for tick in range(20):
        recorder.observe(obs, tick)
        recorder.act(action, tick)
    assert recorder.count == 4
    for tick in (20, 25, 30):  # a frame cut short keeps its action; the 7th frame grows
        recorder.observe(obs, tick)
        recorder.flush(action)
    assert recorder.count == 7 and recorder.images["top"].shape[0] == 12
    assert np.array_equal(recorder.images["top"][6], recorder.latest["top"])
    env.close()


def test_recorder_of_one_episode_is_freed_before_the_next_allocates(tmp_path, monkeypatch):
    import gc
    import weakref

    from tools import collect as module

    alive = []
    original = module.Recorder.__init__

    def tracking_init(self, *args, **kwargs):
        gc.collect()
        assert not any(ref() is not None for ref in alive), "previous recorder still alive"
        alive.append(weakref.ref(self))
        original(self, *args, **kwargs)

    monkeypatch.setattr(module.Recorder, "__init__", tracking_init)
    collect(
        [
            "--target",
            "square",
            "--episodes",
            "2",
            "--steps",
            "10",
            "--policy",
            "policy.py",
            "--keep-failures",
            "--out",
            str(tmp_path),
        ]
    )
    assert len(alive) == 2


def test_teleop_session_finish_saves_and_dead_worker_is_reported(tmp_path, monkeypatch):
    from view import Session

    env = Env("panda", pixels=True)
    env.reset([100000], ["house"])
    session = Session(env, 100000, "house", env.outlines[0], tmp_path / "teleop")
    action = np.r_[env.observe()[0]["qpos"][:7], 1.0]
    for _ in range(6):
        session.before()
        env.step([action])
        session.after(action)
    row = session.finish(action)
    assert row["frames"] == 2 and (tmp_path / "teleop" / row["file"]).exists()
    env.close()
    # A worker that dies without reporting must not hang the collection: the
    # parent notices the exit code, lists the seeds it lost and returns failure.
    dying = tmp_path / "dying.py"
    dying.write_text(
        "import os\nclass Policy:\n    def __init__(self, robot, seed):\n        os._exit(17)\n"
    )
    monkeypatch.setenv("MUJOCO_GL", "egl")
    code = collect(
        [
            "--target",
            "square",
            "--episodes",
            "2",
            "--workers",
            "2",
            "--steps",
            "10",
            "--policy",
            str(dying),
            "--out",
            str(tmp_path / "dead"),
        ]
    )
    assert code == 1 and not list((tmp_path / "dead").rglob("*.npz"))


def test_report_summarizes_a_collection_with_criterion_and_source_digests(tmp_path):
    from tools.report import main as report

    collect(
        [
            "--target",
            "square",
            "--episodes",
            "2",
            "--steps",
            "10",
            "--policy",
            "policy.py",
            "--keep-failures",
            "--out",
            str(tmp_path / "square"),
        ]
    )
    result = report([str(tmp_path / "square"), "--out", str(tmp_path / "summary.json")])
    figure = result["figures"]["square"]
    assert figure["episodes"] == 2 and figure["successes"] == 0 and result["total"] == "0/2"
    assert figure["median_first_success_seconds"] is None
    assert figure["longest_still_seconds"] >= 0 and figure["peak_acceleration"] is not None
    assert result["criterion"]["iou"] == 0.87 and set(result["sources"]) >= {
        "tangram.py",
        "teleop.py",
    }
    assert json.loads((tmp_path / "summary.json").read_text())["figures"]["square"]["episodes"] == 2


def test_each_launch_is_a_dated_round_and_tools_read_every_round(tmp_path, capsys):
    from tools.collect import round_name
    from tools.export import episode_files
    from tools.report import index_rows

    common = ["--target", "square", "--steps", "10", "--policy", "policy.py", "--keep-failures"]
    out = tmp_path / "square"
    collect([*common, "--episodes", "1", "--out", str(out)])
    rounds = [p.name for p in out.iterdir()]
    assert len(rounds) == 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", rounds[0])
    assert rounds[0] == round_name(datetime.strptime(rounds[0], "%Y-%m-%d-%H-%M-%S"))
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == {
        "round": str(out / rounds[0]),
        "recorded": 0,
        "first_seed": 0,
    }
    # A later round of seeds 0-1 (two episodes) and the first round's seed 0:
    # the figure folder yields one file per seed, the newest for seed 0.
    collect([*common, "--episodes", "2", "--out", str(out), "--round", "9999-01-01-00-00-00"])
    files = episode_files(out)
    assert [p.name for p in files] == ["episode-0.npz", "episode-1.npz"]
    assert files[0].parent.name == "9999-01-01-00-00-00"
    rows = index_rows(out)
    assert [r["seed"] for r in rows] == [0, 1] and "9999-01-01-00-00-00" in rows[0]["file"]
    assert [r["seed"] for r in index_rows(out / rounds[0])] == [0]
    from tools.report import report

    summary = report([out / rounds[0]])
    assert list(summary["figures"]) == [f"square/{rounds[0]}"]
    assert summary["figures"][f"square/{rounds[0]}"]["episodes_detail"][0]["round"] == rounds[0]
    # Naming a round again skips the seeds whose file exists.
    collect([*common, "--episodes", "2", "--out", str(out), "--round", "9999-01-01-00-00-00"])
    assert "skipped" in capsys.readouterr().out and len(episode_files(out)) == 2
    # --resume grows the newest round with more seeds, after the ones it lists.
    collect([*common, "--episodes", "2", "--out", str(out), "--resume"])
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first == {"round": str(out / "9999-01-01-00-00-00"), "recorded": 2, "first_seed": 2}
    assert [p.name for p in episode_files(out / "9999-01-01-00-00-00")] == [
        f"episode-{i}.npz" for i in range(4)
    ]
    assert [r["seed"] for r in index_rows(out)] == [0, 1, 2, 3]
    # --retry-failed records only the listed seeds without a file: seed 5 failed
    # (policy.py never solves and the file is not kept), a solving policy is then
    # "the fix", and afterwards nothing is left to retry.
    collect([*common[:-1], "--episodes", "1", "--out", str(out), "--resume", "--offset", "5"])
    assert not (out / "9999-01-01-00-00-00" / "episode-5.npz").exists()
    capsys.readouterr()
    collect([*common, "--out", str(out), "--retry-failed"])
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first["retrying"] == [5] and (out / "9999-01-01-00-00-00" / "episode-5.npz").exists()
    assert [r["seed"] for r in index_rows(out)] == [0, 1, 2, 3, 5]
    collect([*common, "--out", str(out), "--retry-failed"])
    assert json.loads(capsys.readouterr().out.splitlines()[0])["retrying"] == []
    # --offset overrides the start; --resume with no round is an error.
    collect([*common, "--episodes", "1", "--out", str(out), "--resume", "--offset", "7"])
    assert (out / "9999-01-01-00-00-00" / "episode-7.npz").exists()
    with pytest.raises(SystemExit):
        collect([*common, "--episodes", "1", "--out", str(tmp_path / "empty"), "--resume"])
