"""Convert recorded episode folders into one LeRobot v3 dataset.

Usage: uv run -m tools.export data/square data/rectangle data/house
       uv run -m tools.export data/house --push --namespace murobotics

Without `--out` and `--repo-id` the dataset is named after its content,
`tangram-<figures>-<robot>[-subtask]-<N>ep` (for example
`tangram-square-rectangle-house-panda-80ep`), written to `data/lerobot/<name>`
and, on the Hub, `<namespace>/<name>`.

A folder may be a figure folder (every round inside it is read, the newest
copy of a repeated seed wins) or one round. `--push` uploads the dataset to
the Hub under `--repo-id` (public unless `--private`) with a card that
describes the benchmark and the episodes, after `hf auth login`.

Features: observation.images.top and .wrist (video), observation.state (arm
joints and two finger joints) and action (arm joint targets and gripper opening),
all at the recording fps. The figure prompt is the LeRobot task string
(`--task subtask` appends the per-frame annotation instead). Language
annotations are written the way `lerobot-annotate` writes them: the
`language_persistent` column carries `subtask` rows (one per change, stamped
at its start) and `plan` rows (the numbered list of steps still to do, refreshed
at every step boundary), `language_events` stays empty, and both are declared
in meta/info.json. A sidecar episodes.jsonl records seed, silhouette, success,
prompt, plan and the subtask segments per episode index.
Requires the `lerobot` extra: uv sync --extra lerobot
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from env import CAMERAS, HOME, IMAGE_SIZE
from tangram import SCENES, describe_layout, layout


def episode_files(folder):
    """Episode files under a figure folder or one round of it, newest round first
    for a repeated seed (round folders are named by date and time, so they sort)."""
    newest = {}
    for path in sorted(Path(folder).rglob("episode-*.npz")):
        newest[path.name] = path
    return sorted(newest.values())


def scan(folders, include_failures=False):
    """(path, figure, success) per episode file, reading only the small arrays, plus
    the robot: enough to name the dataset before any frame is decoded."""
    items, robot = [], None
    for folder in folders:
        for path in episode_files(folder):
            with np.load(path, allow_pickle=False) as data:
                success = bool(data["success"])
                robot = robot or str(data["robot"])
                if include_failures or success:
                    items.append((path, str(data["target"]), success))
    return items, robot


def dataset_name(items, robot, task="prompt"):
    """`tangram-<figures>-<robot>[-subtask]-<N>ep`, figures in order of appearance."""
    figures = []
    for _, figure, _ in items:
        if figure not in figures:
            figures.append(figure)
    return "-".join(
        [
            "tangram",
            *figures,
            robot or "robot",
            *(["subtask"] if task == "subtask" else []),
            f"{len(items)}ep",
        ]
    )


def episodes(folders, include_failures=False):
    for folder in folders:
        for path in episode_files(folder):
            with np.load(path, allow_pickle=False) as data:
                if not include_failures and not bool(data["success"]):
                    continue
                yield path, {key: data[key] for key in data.files}


def features(robot, narm):
    joints = [f"joint{i}" for i in range(narm)]
    return {
        **{
            f"observation.images.{name}": {
                "dtype": "video",
                "shape": (*IMAGE_SIZE, 3),
                "names": ["height", "width", "channels"],
            }
            for name in CAMERAS
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (narm + 2,),
            "names": [*joints, "finger0", "finger1"],
        },
        "action": {"dtype": "float32", "shape": (narm + 1,), "names": [*joints, "gripper"]},
    }


def segments(subtasks):
    """Change points of a per-frame annotation: [{"frame": i, "text": ...}, ...]."""
    out, last = [], None
    for i, text in enumerate(str(t) for t in subtasks):
        if text != last:
            out.append({"frame": i, "text": text})
            last = text
    return out


def language_rows(data):
    """LeRobot v3.1 persistent language rows for one episode: `subtask` rows at every
    change of the per-frame annotation and, following lerobot-annotate, a `plan` row at
    every step boundary listing the steps still to do as a numbered list."""
    fps = int(data["fps"])
    subtasks = [str(t) for t in data["subtask"]] if "subtask" in data else []
    steps = [int(t) for t in data["step"]] if "step" in data else []
    plan = [str(t) for t in data["plan"]] if "plan" in data else []
    rows = []
    for segment in segments(subtasks):
        if segment["text"]:
            rows.append(
                {
                    "role": "assistant",
                    "content": segment["text"],
                    "style": "subtask",
                    "timestamp": segment["frame"] / fps,
                    "camera": None,
                    "tool_calls": None,
                }
            )
    last = None
    for frame, step in enumerate(steps):
        if step and step != last and plan:
            remaining = plan[step - 1 :]
            rows.append(
                {
                    "role": "assistant",
                    "content": "\n".join(f"{k}. {line}" for k, line in enumerate(remaining, 1)),
                    "style": "plan",
                    "timestamp": frame / fps,
                    "camera": None,
                    "tool_calls": None,
                }
            )
            last = step
    return sorted(rows, key=lambda r: (r["timestamp"], r["style"]))


def annotate(out, rows_by_episode):
    """Add the two LeRobot language columns to every data shard and declare them in info."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.io_utils import (
        load_info,
        write_info,
        write_table_one_row_group_per_episode,
    )
    from lerobot.datasets.language import (
        LANGUAGE_EVENTS,
        LANGUAGE_PERSISTENT,
        language_feature_info,
    )

    for path in sorted((Path(out) / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        # Like lerobot-annotate's writer, let pyarrow infer the struct type: the canonical
        # type's JSON extension for tool_calls cannot be built from Python lists.
        persistent = pa.array([rows_by_episode.get(e, []) for e in episodes])
        events = pa.array([[] for _ in episodes])
        table = table.append_column(LANGUAGE_PERSISTENT, persistent)
        table = table.append_column(LANGUAGE_EVENTS, events)
        write_table_one_row_group_per_episode(table, path)
    info = load_info(Path(out))
    info.features = {**info.features, **language_feature_info()}
    write_info(info, Path(out))


def export(folders, out, repo_id, include_failures=False, overwrite=False, task="prompt"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    dataset, rows, fps, robot = None, [], None, None
    language = {}
    for path, data in episodes(folders, include_failures):
        if dataset is None:
            fps, robot = int(data["fps"]), str(data["robot"])
            # Frames go straight to the encoder threads: no PNG round trip through disk,
            # which otherwise costs more than the encoding itself.
            dataset = LeRobotDataset.create(
                repo_id,
                fps,
                features(robot, len(HOME[robot])),
                root=out,
                robot_type=robot,
                streaming_encoding=True,
            )
        if int(data["fps"]) != fps or str(data["robot"]) != robot:
            raise ValueError(f"{path}: mixed fps or robot within one dataset")
        subtasks = data["subtask"] if "subtask" in data else [""] * len(data["time"])
        for t in range(len(data["time"])):
            text = str(data["prompt"])
            if task == "subtask" and str(subtasks[t]):
                text = f"{text} {subtasks[t]}"
            dataset.add_frame(
                {
                    **{f"observation.images.{name}": data[f"images_{name}"][t] for name in CAMERAS},
                    "observation.state": data["state"][t],
                    "action": data["action"][t],
                    "task": text,
                }
            )
        dataset.save_episode()
        language[len(rows)] = language_rows(data)
        rows.append(
            {
                "episode_index": len(rows),
                "source": str(path),
                "seed": int(data["seed"]),
                "target": str(data["target"]),
                "success": bool(data["success"]),
                "frames": int(len(data["time"])),
                "prompt": str(data["prompt"]),
                "plan": [str(t) for t in data["plan"]] if "plan" in data else [],
                "subtasks": segments(subtasks),
            }
        )
    if dataset is None:
        raise ValueError("No episodes found; record some with tools/collect.py")
    dataset.finalize()
    annotate(out, language)
    with (out / "episodes.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


GITHUB = "https://github.com/murobotics-ai/tangram-bench"


def card(rows, robot, fps, task):
    """Markdown for the card's description: a summary table, one row per figure,
    how the episodes were recorded and what each field holds."""
    figures = {}
    for row in rows:
        f = figures.setdefault(
            row["target"], {"episodes": 0, "solved": 0, "frames": 0, "prompt": ""}
        )
        f["episodes"] += 1
        f["solved"] += int(row["success"])
        f["frames"] += int(row["frames"])
        f["prompt"] = f["prompt"] or row["prompt"]
    episodes, solved = len(rows), sum(r["success"] for r in rows)
    frames = sum(r["frames"] for r in rows)
    minutes = frames / fps / 60
    scenes = len(
        {
            (
                r["target"],
                json.dumps(describe_layout(layout(r["seed"], r["target"])), sort_keys=True),
            )
            for r in rows
        }
    )
    lines = [
        f"Demonstrations for [Tangram-Bench]({GITHUB}): a {robot.capitalize()} arm in MuJoCo "
        "assembles a tangram silhouette from a packed square of seven pieces, given the "
        "silhouette outline on the table and a one-sentence prompt.",
        "",
        "| | |",
        "|---|---|",
        f"| Episodes | {episodes} ({solved} solved) |",
        f"| Distinct scenes | {scenes}; the designed grid holds {SCENES:,} per figure, "
        f"seed n + {SCENES} repeating seed n |",
        f"| Frames | {frames:,} at {fps} fps, {minutes:.0f} min of manipulation |",
        f"| Figures | {', '.join(figures)} |",
        f"| Robot | {robot}, simulated (MuJoCo) |",
        "| Cameras | `observation.images.top`, `observation.images.wrist`, 320x240 |",
        "| State | `observation.state`: 7 arm joints and 2 finger joints, rad and m |",
        "| Action | `action`: 7 joint targets in rad and a gripper command, 1 open, 0 closed |",
        "| Task | the figure prompt"
        + (" followed by the current subtask" if task == "subtask" else "")
        + " |",
        "| Demonstrator | the repository's reference controller, one seed per episode |",
        "",
        "| Figure | Episodes | Solved | Prompt |",
        "|---|---|---|---|",
        *(
            f"| {name} | {f['episodes']} | {f['solved']} | {f['prompt']} |"
            for name, f in figures.items()
        ),
        "",
        "### How the episodes were recorded",
        "",
        "Scenes come from the benchmark's designed grid: goal yaw every 30 degrees, packed "
        "square yaw every 45 degrees, centres offset by up to 30 mm, all inside the arm's "
        "workspace. A seed fixes the scene and the demonstrator, so any episode reproduces "
        "exactly with `tools.collect` in the repository; `episodes.jsonl` maps each "
        "episode index to its seed, figure, success, prompt, plan and subtask segments. An "
        "episode ends four seconds after the benchmark's success test has held, once the "
        "arm has returned home.",
        "",
        "### Language annotations",
        "",
        "`language_persistent` carries the demonstrator's numbered plan (refreshed at every "
        "step) and the subtask sentence in progress, in the layout `lerobot-annotate` "
        "writes; `language_events` is empty.",
        "",
    ]
    return "\n".join(lines)


def push(out, repo_id, rows, task="prompt", private=False):
    """Upload the dataset at `out` to the Hub as `repo_id` with a descriptive card."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=Path(out))
    dataset.push_to_hub(
        tags=["tangram", "mujoco", "franka", "simulation", "manipulation"],
        license="mit",
        private=private,
        dataset_description=card(rows, dataset.meta.robot_type, dataset.fps, task),
        url=GITHUB,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+", type=Path, help="Episode folders from tools/collect.py")
    p.add_argument(
        "--out",
        type=Path,
        help="LeRobot dataset root; default data/lerobot/<name>, with <name> "
        "tangram-<figures>-<robot>[-subtask]-<N>ep from the episodes themselves",
    )
    p.add_argument("--repo-id", help="Dataset id on the Hub; default <namespace>/<name>")
    p.add_argument(
        "--namespace",
        help="Hub user or organisation for the default --repo-id; with --push, default the "
        "logged-in user",
    )
    p.add_argument("--include-failures", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--push",
        action="store_true",
        help="Upload to the Hub as --repo-id; with an existing --out and no --overwrite, "
        "upload that dataset as it is",
    )
    p.add_argument("--private", action="store_true", help="With --push: private repository")
    p.add_argument(
        "--task",
        choices=["prompt", "subtask"],
        default="prompt",
        help="LeRobot task string: the figure prompt alone (what a policy sees at evaluation) "
        "or the prompt followed by the per-frame subtask, for subtask-conditioned training",
    )
    args = p.parse_args(argv)
    items, robot = scan(args.folders, args.include_failures)
    name = dataset_name(items, robot, args.task)
    args.out = args.out or Path("data") / "lerobot" / name
    if args.repo_id is None:
        namespace = args.namespace
        if namespace is None and args.push:
            from huggingface_hub import HfApi

            namespace = HfApi().whoami()["name"]
        args.repo_id = f"{namespace}/{name}" if namespace else name
    if args.push and args.out.exists() and not args.overwrite:
        # Push what an earlier export left here (its episodes.jsonl carries the rows).
        rows = [json.loads(line) for line in (args.out / "episodes.jsonl").read_text().splitlines()]
    else:
        rows = export(
            args.folders, args.out, args.repo_id, args.include_failures, args.overwrite, args.task
        )
    counts = {}
    for row in rows:
        counts[row["target"]] = counts.get(row["target"], 0) + 1
    summary = {
        "episodes": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "per_target": counts,
        "out": str(args.out),
        "repo_id": args.repo_id,
    }
    if args.push:
        summary["url"] = push(args.out, args.repo_id, rows, args.task, args.private)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
