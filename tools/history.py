"""Build a local, self-contained checkpoint history and episode viewer from run outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark import COMPARISON_FIELDS
from tangram import VERTICES
from tools.results import verify


def collect(directory, stride=10):
    runs = []
    for path in sorted(directory.rglob("*.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict) or "finished_at" not in result or "episodes" not in result:
            continue
        cohort = {key: result.get(key) for key in COMPARISON_FIELDS}
        cohort["seeds"] = result.get("seeds")
        group = hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()[:12]
        record = {
            key: result.get(key)
            for key in (
                "started_at",
                "robot",
                "protocol",
                "policy",
                "policy_sha256",
                "policy_artifacts",
                "status",
                "summary",
                "prompt",
                "backend",
                "policy_metadata",
            )
        }
        system = (result.get("policy_metadata") or {}).get("system", {})
        checkpoint = system.get("checkpoint")
        record["system_label"] = system.get("model") or Path(result.get("policy", "")).name
        record["checkpoint_sha256"] = (result.get("policy_artifacts") or {}).get(checkpoint)
        record.update(cohort=group, file=str(path.resolve()), episodes=[])
        try:
            verify(path)
            record["verification"] = "verified"
        except (ValueError, KeyError, OSError):
            record["verification"] = "unverified / different source version"
        for row in result["episodes"]:
            episode = dict(row)
            # Only deserialize trajectory data for successfully verified runs.
            if record["verification"] == "verified":
                with np.load(path.parent / row["trajectory"], allow_pickle=False) as trace:
                    indices = np.unique(
                        np.r_[np.arange(0, len(trace["time"]), stride), len(trace["time"]) - 1]
                    )
                    episode["poses"] = trace["pieces"][indices].tolist()
                    episode["time"] = trace["time"][indices].tolist()
                    episode["goal"] = trace["goal"][0].tolist()
            record["episodes"].append(episode)
        runs.append(record)
    return sorted(runs, key=lambda r: r["started_at"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/runs/history.html"))
    p.add_argument(
        "--stride", type=int, default=10, help="Display every Nth frame; scoring is unchanged"
    )
    args = p.parse_args()
    if args.stride < 1:
        p.error("stride must be positive")
    runs = collect(args.directory, args.stride)
    if not runs:
        p.error("No evaluation results found")
    data = json.dumps(
        {"runs": runs, "vertices": [v.tolist() for v in VERTICES], "stride": args.stride},
        allow_nan=False,
    ).replace("<", "\\u003c")
    template = Path(__file__).with_name("history.html").read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(template.replace("__RUN_DATA__", data))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
