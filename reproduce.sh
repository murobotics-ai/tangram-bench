#!/usr/bin/env bash
# The whole benchmark, end to end, on one laptop with an 8 GB GPU.
# Wall time: about 30 min of demonstrations with 10 workers, 2-4 h of fine-tuning, 30 min of evaluation.
# Run it detached and keep the log:  screen -L -Logfile reproduce.log -S tangram bash reproduce.sh
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl}

uv sync --frozen --extra dev --extra lerobot
uv run -m tools.prepare

# 1. Demonstrations: 60 seeds per training figure, ten arms at a time; nearly all succeed.
for figure in square rectangle house; do
  uv run -m tools.collect --target "$figure" --episodes 60 --workers 10 --out "data/$figure"
done
uv run -m tools.export data/square data/rectangle data/house --out data/lerobot/train

# 2. Fine-tune SmolVLA: weights download to checkpoints/, the run goes to
#    outputs/train/smolvla (batch 4 fits in 8 GB; add --batch-size 2 if nvidia-smi disagrees).
uv run -m tools.train --policy smolvla --steps 20000

# 3. Evaluate: seen figures at unseen poses (dev) and the held-out figure (test).
uv run eval.py --policy policy.py --split dev --episodes 30 --out outputs/runs/hold-dev.json
uv run eval.py --system examples/systems/lerobot.json --obs pixels --max-chunk 50 \
  --split dev --episodes 30 --out outputs/runs/smolvla-dev.json
uv run eval.py --system examples/systems/lerobot.json --obs pixels --max-chunk 50 \
  --split test --episodes 10 --out outputs/runs/smolvla-test.json
uv run -m tools.results compare outputs/runs/hold-dev.json outputs/runs/smolvla-dev.json
