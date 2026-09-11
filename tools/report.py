"""Summarize demonstration folders: success, smoothness and stillness per figure.

Usage: uv run -m tools.report data/square data/rectangle data/house data/cat
       uv run -m tools.report data/house --out results/2026-09-11-oracle-panda.json

Reads the `index.jsonl` and episode files that tools/collect.py writes. Success
is the collector's: the benchmark's success test held for HOLD_STEPS during the
episode (episodes then stop four seconds later, once the arm is home), which is not the same as
success at the full horizon; eval.py measures that. The JSON records the
success thresholds and digests of the scoring, controller and IK sources, so a
number can always be traced to the code that produced it.

Definitions, per episode, from the 10 fps `tcp_pos` and `state` arrays:
- tilted hand: any frame with the tool axis more than 60 degrees from down;
- TCP above 0.5 m after the first 30 s;
- longest still interval: consecutive frames with the TCP under 3 mm/s;
- peak speed and peak acceleration of the TCP, the latter from the velocity
  vector (finite differences at the frame rate), so turns count.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import tangram
from benchmark import CONTROL_SECONDS, digest, wilson_interval
from env import make_model

SOURCES = ("tangram.py", "examples/oracle.py", "teleop.py", "env.py")
CRITERION = {
    "iou": tangram.IOU_THRESHOLD,
    "overlap": tangram.OVERLAP_THRESHOLD,
    "piece_coverage": tangram.PIECE_COVERAGE,
    "table_tolerance_m": tangram.TABLE_TOLERANCE,
    "flat_degrees": tangram.FLAT_DEGREES,
}


def episode_metrics(path, model, data):
    with np.load(path, allow_pickle=False) as d:
        state, tcp, pieces, goal = d["state"], d["tcp_pos"].astype(float), d["pieces"], d["goal"]
        fps = int(d["fps"])
        first = int(d["first_success_step"]) if "first_success_step" in d.files else -1
    final = tangram.score(pieces[-1], np.zeros((7, 6)), goal)
    tilted = False
    for frame in state:
        data.qpos[: len(frame)] = frame
        mujoco.mj_forward(model, data)
        if data.site("tcp").xmat.reshape(3, 3)[2, 2] > -0.5:
            tilted = True
            break
    velocity = np.diff(tcp, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=1)
    run = longest = 0
    for still in speed < 0.003:
        run = run + 1 if still else 0
        longest = max(longest, run)
    return {
        "iou": round(float(final["iou"]), 4),
        "pieces_in_goal": int(final["pieces_in_goal"]),
        "tilted_hand": bool(tilted),
        "tcp_above_half_metre": bool((tcp[30 * fps :, 2] > 0.5).any()),
        "longest_still_seconds": round(longest / fps, 1),
        "peak_speed": round(float(speed.max()), 3) if len(speed) else 0.0,
        "peak_acceleration": round(
            float(np.linalg.norm(np.diff(velocity, axis=0), axis=1).max() * fps), 2
        )
        if len(velocity) > 1
        else 0.0,
        "first_success_seconds": round(first * CONTROL_SECONDS, 1) if first >= 0 else None,
    }


def summarize(folder, model, data):
    rows = [json.loads(line) for line in (folder / "index.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: r["seed"])
    episodes = []
    for row in rows:
        entry = {
            "seed": row["seed"],
            "success": bool(row["success"]),
            "status": row["status"],
            "error": row["error"]["message"] if row.get("error") else None,
        }
        if row.get("file"):
            entry.update(episode_metrics(folder / row["file"], model, data))
        episodes.append(entry)
    successes = [e for e in episodes if e["success"]]
    times = [e["first_success_seconds"] for e in successes if e.get("first_success_seconds")]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "ci95": list(wilson_interval(len(successes), len(episodes))) if episodes else None,
        "iou_of_successes": [min(e["iou"] for e in successes), max(e["iou"] for e in successes)]
        if successes
        else None,
        "median_first_success_seconds": float(np.median(times)) if times else None,
        "tilted_hand_episodes": sum(e.get("tilted_hand", False) for e in episodes),
        "tcp_above_half_metre_episodes": sum(
            e.get("tcp_above_half_metre", False) for e in episodes
        ),
        "longest_still_seconds": max(
            (e.get("longest_still_seconds", 0) for e in episodes), default=0
        ),
        "mean_peak_speed": float(np.mean([e["peak_speed"] for e in episodes if "peak_speed" in e]))
        if any("peak_speed" in e for e in episodes)
        else None,
        "peak_acceleration": {
            "mean_of_episode_peaks": float(
                np.mean([e["peak_acceleration"] for e in episodes if "peak_acceleration" in e])
            ),
            "max": float(max(e["peak_acceleration"] for e in episodes if "peak_acceleration" in e)),
        }
        if any("peak_acceleration" in e for e in episodes)
        else None,
        "episodes_detail": episodes,
    }


def report(folders, robot="panda"):
    model = make_model(robot)
    data = mujoco.MjData(model)
    figures = {}
    for folder in folders:
        folder = Path(folder)
        figures[folder.name] = summarize(folder, model, data)
    root = Path(__file__).resolve().parents[1]
    return {
        "criterion": CRITERION,
        "sources": {name: digest(root / name) for name in SOURCES},
        "collector": "tools/collect.py; success = the success test held for HOLD_STEPS, then four more seconds",
        "total": f"{sum(f['successes'] for f in figures.values())}/{sum(f['episodes'] for f in figures.values())}",
        "figures": figures,
    }


def table(result):
    lines = []
    for name, f in result["figures"].items():
        ci = f["ci95"] or [0, 0]
        iou = f["iou_of_successes"]
        median = f["median_first_success_seconds"]
        accel = f["peak_acceleration"] or {"mean_of_episode_peaks": 0, "max": 0}
        lines.append(
            f"{name:10s} success {f['successes']}/{f['episodes']}  wilson {ci[0]:.2f}-{ci[1]:.2f}  "
            f"iou {iou[0]:.3f}-{iou[1]:.3f}  "
            if iou
            else f"{name:10s} success 0/{f['episodes']}  "
        )
        lines[-1] += (
            f"median first success {median:.0f} s  " if median else "median first success -  "
        ) + (
            f"tilted {f['tilted_hand_episodes']}  tcp>0.5m {f['tcp_above_half_metre_episodes']}  "
            f"longest still {f['longest_still_seconds']:.1f} s  "
            f"accel mean {accel['mean_of_episode_peaks']:.1f} max {accel['max']:.1f} m/s2"
        )
        for e in f["episodes_detail"]:
            if e["error"]:
                lines.append(f"    error seed {e['seed']}: {e['error'][:80]}")
    lines.append(f"TOTAL {result['total']}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+", type=Path, help="Episode folders, one per figure")
    p.add_argument("--robot", default="panda")
    p.add_argument("--out", type=Path, help="Write the full JSON summary here")
    p.add_argument("--note", default="", help="Free text stored in the JSON")
    args = p.parse_args(argv)
    result = report(args.folders, args.robot)
    if args.note:
        result["note"] = args.note
    print(table(result))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(args.out)
    return result


if __name__ == "__main__":
    main()
