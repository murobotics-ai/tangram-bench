"""Rewrite an exported dataset's language annotations in the index format.

Usage: uv run -m tools.reindex data/lerobot/tangram-square-rectangle-house-panda-300ep
       uv run -m tools.reindex <root> --relabel   # also rewrite the legacy sentences

Datasets exported before 12 Sep 2026 carried the plan and subtask annotations in
LeRobot's `language_persistent` column, the whole list of rows repeated on every
frame. This rewrites such a dataset in place into the format `tools.export`
writes now: a `subtask_index` and a `plan_index` per frame pointing into
meta/subtasks.parquet and meta/plans.parquet, the language columns dropped,
meta/info.json, meta/stats.json and the per-episode stats updated. The
annotations come from the dataset's own episodes.jsonl sidecar, so no raw
episode is needed. Running it on a dataset already in the index format
rebuilds the same files. Videos are untouched.

`--relabel` first rewrites the sidecar's sentences from the vocabulary the
oracle used before 13 Sep 2026 (a sentence per motion: reach, lift, carry,
lower, release, with the figure's name in it) to the current one: the plan
line of the piece being placed, `Place the <piece> at the <place> of the
outline.`, the recovery sentences unchanged. Consecutive equal segments merge.
Requires the `lerobot` extra: uv sync --extra lerobot
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools.export import INDEX_FEATURE, VOCABULARIES, Vocabulary, plan_text

INDEX_COLUMNS = tuple(f"{name}_index" for name in VOCABULARIES)
LEGACY_MOTION = re.compile(
    r"^(?:Reach for|Lift|Carry|Lower|Release) the (.+?)(?: and| off| to| into)\b"
)
LEGACY_PLAN = re.compile(r"^Place the (.+?) at the (.+?) of the \w+\.$")


def relabel(row):
    """The episodes.jsonl row with its legacy motion sentences replaced by the plan
    line of the piece in motion, and the plan lines' figure name replaced by `outline`."""
    lines = {}
    for line in row["plan"]:
        piece, where = LEGACY_PLAN.match(line).groups()
        lines[piece] = f"Place the {piece} at the {where} of the outline."
    segments = []
    for segment in row["subtasks"]:
        motion = LEGACY_MOTION.match(segment["text"])
        text = lines[motion.group(1)] if motion else segment["text"]
        if not segments or segments[-1]["text"] != text:
            segments.append({"frame": segment["frame"], "text": text})
    return {**row, "plan": list(lines.values()), "subtasks": segments}


def relabel_sidecar(root):
    path = Path(root) / "episodes.jsonl"
    rows = [relabel(json.loads(line)) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def annotations(root):
    """Per episode index: the plan text and the subtask text of every frame, from episodes.jsonl."""
    out = {}
    for line in (Path(root) / "episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        subtasks = [""] * row["frames"]
        for segment in row["subtasks"]:
            subtasks[segment["frame"] :] = [segment["text"]] * (row["frames"] - segment["frame"])
        out[row["episode_index"]] = (plan_text(row["plan"]), subtasks)
    return out


def index_arrays(episodes, vocabularies):
    """Per episode index: {column: int64 array over its frames}, registering the texts
    in episode order so indices are stable across runs."""
    out = {}
    for episode, (plan, subtasks) in sorted(episodes.items()):
        plan_index = vocabularies["plan"](plan)
        out[episode] = {
            "subtask_index": np.array(
                [vocabularies["subtask"](t) for t in subtasks], dtype=np.int64
            ),
            "plan_index": np.full(len(subtasks), plan_index, dtype=np.int64),
        }
    return out


def reindex(root):
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import (
        load_info,
        load_stats,
        write_info,
        write_stats,
        write_table_one_row_group_per_episode,
    )
    from lerobot.datasets.language import LANGUAGE_COLUMNS
    from lerobot.utils.constants import DEFAULT_FEATURES

    root = Path(root)
    vocabularies = {name: Vocabulary(name) for name in VOCABULARIES}
    arrays = index_arrays(annotations(root), vocabularies)
    index_features = {column: INDEX_FEATURE for column in INDEX_COLUMNS}

    info = load_info(root)
    kept = {
        k: v
        for k, v in info.features.items()
        if k not in LANGUAGE_COLUMNS and k not in INDEX_COLUMNS and k not in DEFAULT_FEATURES
    }
    info.features = {
        **kept,
        **index_features,
        **{k: v for k, v in info.features.items() if k in DEFAULT_FEATURES},
    }
    schema = get_hf_features_from_features(info.features).arrow_schema

    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        table = table.drop_columns(
            [c for c in table.column_names if c in LANGUAGE_COLUMNS + INDEX_COLUMNS]
        )
        episodes = table.column("episode_index").to_numpy()
        frames = table.column("frame_index").to_numpy()
        for column in INDEX_COLUMNS:
            values = np.array([arrays[e][column][f] for e, f in zip(episodes, frames, strict=True)])
            table = table.append_column(column, pa.array(values, type=pa.int64()))
        table = table.select(schema.names).replace_schema_metadata(schema.metadata)
        write_table_one_row_group_per_episode(table, path)

    stats_per_episode = {
        episode: compute_episode_stats(
            {column: values[column].reshape(-1, 1) for column in INDEX_COLUMNS}, index_features
        )
        for episode, values in arrays.items()
    }
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        frame = pd.read_parquet(path)
        for column in INDEX_COLUMNS:
            for stat in next(iter(stats_per_episode.values()))[column]:
                frame[f"stats/{column}/{stat}"] = [
                    stats_per_episode[e][column][stat].tolist() for e in frame["episode_index"]
                ]
        frame = frame.drop(
            columns=[
                c
                for c in frame.columns
                if c.split("/")[1:2] and c.split("/")[1] in LANGUAGE_COLUMNS
            ]
        )
        frame.to_parquet(path)

    stats = {
        k: v
        for k, v in (load_stats(root) or {}).items()
        if k not in LANGUAGE_COLUMNS + INDEX_COLUMNS
    }
    stats.update(aggregate_stats(list(stats_per_episode.values())))
    write_stats(stats, root)
    write_info(info, root)
    for vocabulary in vocabularies.values():
        vocabulary.write(root)
    return {name: len(vocabulary.index) for name, vocabulary in vocabularies.items()}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root", type=Path, help="Dataset root written by tools/export.py")
    p.add_argument(
        "--relabel",
        action="store_true",
        help="Rewrite the sidecar's pre-13 Sep 2026 motion sentences to the plan-line vocabulary first",
    )
    args = p.parse_args(argv)
    if args.relabel:
        relabel_sidecar(args.root)
    print(json.dumps({"root": str(args.root), **reindex(args.root)}))


if __name__ == "__main__":
    main()
