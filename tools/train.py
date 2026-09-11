"""Fine-tune a LeRobot policy on an exported dataset: a thin front for lerobot-train.

Usage: uv run -m tools.train --policy smolvla                  # 20k steps, batch 4
       uv run -m tools.train --policy act --steps 50000 --out outputs/train/act
       uv run -m tools.train --policy smolvla --dry-run        # print the command only
Any other argument goes to lerobot-train unchanged, e.g. --batch_size=2 --resume=true.

Pretrained weights (lerobot/smolvla_base and its SmolVLM2 backbone) download to
checkpoints/ on first use; set HF_HUB_CACHE to keep them elsewhere. A run
writes outputs/train/NAME/checkpoints/last/pretrained_model, the directory that
examples/systems/lerobot.json loads. lerobot-train refuses an existing output
directory unless --resume=true is given.
Requires the `lerobot` extra: uv sync --extra lerobot
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
POLICIES = {
    "smolvla": [
        "--policy.path=lerobot/smolvla_base",
        "--policy.input_features=null",  # camera and state names come from the dataset
        "--policy.output_features=null",
    ],
    "act": ["--policy.type=act"],
}
SCHEDULED = {"smolvla"}  # policies whose learning-rate schedule takes a decay length


def command(
    policy,
    dataset="data/lerobot/train",
    out=None,
    steps=20000,
    batch_size=4,
    n_action_steps=10,
    device="cuda",
    repo_id="tangram-bench/local",
    extra=(),
):
    """The lerobot-train argument list; defaults measured on an 8 GB GPU."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {sorted(POLICIES)}")
    out = out or f"outputs/train/{policy}"
    schedule = [f"--policy.scheduler_decay_steps={steps}"] if policy in SCHEDULED else []
    return [
        "lerobot-train",
        *POLICIES[policy],
        *schedule,
        "--policy.push_to_hub=false",
        f"--policy.device={device}",
        f"--policy.n_action_steps={n_action_steps}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--wandb.enable=false",
        f"--output_dir={out}",
        *extra,
    ]


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--policy", default="smolvla", choices=sorted(POLICIES))
    p.add_argument("--dataset", default="data/lerobot/train", help="Exported LeRobot dataset")
    p.add_argument("--out", help="Run directory; default outputs/train/<policy>")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=4, help="4 peaks at 2.4 GB for SmolVLA")
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="Actions per chunk; 10 = one second at 10 fps",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--repo-id", default="tangram-bench/local", help="Name stored with the run")
    p.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args, extra = p.parse_known_args(argv)
    argv = command(
        args.policy,
        args.dataset,
        args.out,
        args.steps,
        args.batch_size,
        args.n_action_steps,
        args.device,
        args.repo_id,
        extra,
    )
    env = dict(os.environ)
    env.setdefault("HF_HUB_CACHE", str(CHECKPOINTS))
    print(f"HF_HUB_CACHE={env['HF_HUB_CACHE']} {shlex.join(argv)}")
    if args.dry_run:
        return 0
    exe = Path(sys.executable).with_name(argv[0])
    if not exe.exists():
        p.error("lerobot-train not installed; run: uv sync --extra dev --extra lerobot")
    Path(env["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    return subprocess.run([str(exe), *argv[1:]], env=env, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
