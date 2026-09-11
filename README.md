<p align="center">
  <img src="assets/preview.png" alt="Packed tangram square, a Franka Panda and the target shadow on the table" width="760">
</p>

<h1 align="center">Tangram-Bench</h1>

<p align="center"><b>Seven pieces, one arm, one silhouette. Does the policy generalize to a shape it never saw?</b></p>

<p align="center">
  <a href="https://github.com/murobotics-ai/tangram-bench/actions/workflows/test.yml"><img src="https://github.com/murobotics-ai/tangram-bench/actions/workflows/test.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/status-prototype-orange" alt="status: prototype">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="python 3.12+">
  <img src="https://img.shields.io/badge/physics-MuJoCo-2c3e50" alt="MuJoCo">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

Tangram-Bench is a small, readable robot-manipulation benchmark that asks one
question and measures it exactly: **can a policy rebuild a tangram silhouette it
was never trained on?** A Franka Panda starts with the seven pieces packed as a
square, sees the target silhouette painted on the table, and has to assemble it.
Success is a global constraint over all seven pieces, scored by exact geometry.
Everything runs in simulation on one laptop with an 8 GB GPU, in about 4,000 lines
of plain Python: record demonstrations, train a policy, evaluate it on seen and
unseen shapes, and compare results by seed. The current headline is the
reference controller at **15 of 40 assemblies**; the first learned baseline is
pending (see [Results](#results)).

## The problem in one minute

- **Pieces.** The classic seven-piece dissection, 5 mm slabs with a square knob
  on each centroid. The knob is the only intended grasp, so the benchmark
  measures *where each piece goes*, not how to pinch a thin slab off a table.
- **Start.** The pieces are packed as a square at a random position and yaw.
- **Goal.** A silhouette drawn on the table at another random pose: square,
  rectangle or the classic tangram house (body, roof, chimney) for training, and
  the **cat**, which no policy trains on. The policy also gets a text prompt.
- **Observation.** Either exact state (joint angles, piece poses, goal outline)
  or two cameras, top and wrist, at 320×240. The contract is recorded with every
  result.
- **Language.** The prompt names the figure: *Solve the tangram puzzle to
  assemble the house.* Demonstrations also carry a numbered plan, one step per
  piece, and a per-frame subtask sentence, the kind of label many VLAs train
  on: *SUBTASK 3/7: Carry the orange large triangle to the right of the house
  and align it.* Exported datasets store them in LeRobot's own language
  columns. Policies never see them at evaluation; the replay viewer shows the
  label changing over time.
- **Action.** Absolute joint targets in radians plus gripper opening in [0, 1],
  at 50 Hz, one action or a chunk per call.
- **Success.** Silhouette IoU ≥ 0.95, overlap under 1%, every piece flat, on the
  table and still for half a second. The inset pieces cap IoU at 0.979, so
  success means every piece within about a millimetre.
- **Splits.** *Train* seeds feed demonstrations; *dev* is the training figures at
  unseen poses; *test* is the cat. Reporting dev and test side by side is the
  point: the gap is the generalization result.

Why tangram: most manipulation suites check a per-object condition and can be
matched by models that ignore language or collapse under pose randomization
([Jiang et al., 2026](https://arxiv.org/abs/2606.04233)). Here the target is a
silhouette, not a list of poses, so the policy must infer an assignment before any
control problem exists; each placed piece constrains the rest; several
decompositions tile the same outline and all of them score; and millimetre
precision means control errors show up directly. Frontier vision-language models
still struggle with the planning half alone ([TangramSR,
2026](https://arxiv.org/abs/2602.05570)), which is why the benchmark also has an
API track for language models.

Two caveats, stated plainly. Four public figures and one held-out animal do not
establish broad shape-family generalization; a generated corpus with structural
families is the next milestone in [docs/plan.md](docs/plan.md). And precise
placement of thin slabs may dominate over reasoning; the reference controller's
failure modes below say how much.

## Quick start

Install [uv](https://docs.astral.sh/uv/) and Git, then:

```bash
uv sync --frozen --extra dev
uv run -m tools.prepare                 # robot assets, pinned upstream revision
uv run view.py --target house           # look at the scene
```

`bash reproduce.sh` runs the whole pipeline below unattended: about two hours of
demonstrations, two to four hours of fine-tuning and half an hour of evaluation.

## The pipeline

```bash
uv run view.py --teleop --record data/house-teleop        # 1. drive the arm, record demos
uv run -m tools.collect --target house --episodes 60      # 1'. scripted demos for one figure
uv run -m tools.export data/square data/rectangle data/house --out data/lerobot/train
uv run -m tools.train --policy smolvla                    # 2. fine-tune (lerobot-train)
uv run eval.py --system examples/systems/lerobot.json --obs pixels --max-chunk 50 \
    --split dev --episodes 30 --out outputs/runs/smolvla-dev.json   # 3. evaluate
uv run -m tools.results compare outputs/runs/hold-dev.json outputs/runs/smolvla-dev.json
```

Everything generated lands in three git-ignored folders, so the source tree stays
the source tree:

```text
data/<figure>/             demonstrations, one npz per episode     tools.collect, view.py --record
data/lerobot/train/        exported LeRobot dataset                tools.export
checkpoints/               pretrained weights from the Hub         tools.train, first run
outputs/train/<name>/      fine-tuning runs; checkpoints/last/pretrained_model is what eval loads
outputs/runs/<name>.json   evaluation results; trajectories in <name>.artifacts/   eval.py
results/                   versioned summaries and the experiment log (committed)
```

### 1. Demonstrations

`tools/collect.py` takes the silhouette as input, runs train-split seeds with a
policy (default: the reference controller below) and writes one compressed npz
per episode: top and wrist images, `qpos` as state, the commanded action, piece
poses, goal, the figure prompt and the per-frame `subtask` annotation the
demonstrator exposes, at `--fps` frames per second (default 10). Replay one with
`uv run view.py --demo data/house/episode-3.npz` and watch the annotation change. A frame's
action is the last command of its 50/fps-tick interval, so a policy that predicts
one action per frame and holds it for the interval reproduces the demonstration.
Episodes stop one second after a held success. Failed attempts are listed in
`index.jsonl` and skipped unless `--keep-failures`.

Teleoperation records the same format. The keyboard stands in for a SpaceMouse:
**W/S A/D Q/E** translate, **Z/X T/G C/V** rotate, **R/F** open and close the
gripper while held, **O** resets, **Shift** is fine motion, **P** pauses, **1/2/3**
switch the main camera, **Esc** exits. Every episode ended by **O** or **Esc** is
saved, and the overlay shows `SOLVED` once the assembly has held.

`tools/export.py` converts folders to one LeRobot v3 dataset:
`observation.images.top`, `observation.images.wrist` (video),
`observation.state` (9: arm joints and fingers), `action` (8: joint targets and
gripper), the figure prompt as task (`--task subtask` appends the annotation
for subtask-conditioned training), the plan and subtask annotations in
LeRobot's `language_persistent` column exactly as `lerobot-annotate` writes
them, and an `episodes.jsonl` sidecar with seed, silhouette, success, prompt,
plan and the annotation segments per episode. Needs the `lerobot` extra:
`uv sync --extra dev --extra lerobot`.

### 2. Train a policy

The first baseline is **SmolVLA** (LeRobot, ~450 M parameters): the only
vision-language-action model that fine-tunes on an 8 GB laptop GPU. Measured here:
batch 4 peaks at 2.4 GB. ACT is the cheap non-language control. π0.5 needs 24 GB or
more to fine-tune and about 8 GB of weights at inference, so it is a cloud option
that the same wrapper can run. Sources and numbers: [docs/plan.md](docs/plan.md).

```bash
uv run -m tools.train --policy smolvla --steps 20000      # add --dry-run to see the command
```

`tools/train.py` is a thin front for `lerobot-train`: it downloads
`lerobot/smolvla_base` and its SmolVLM2 backbone to `checkpoints/`, reads
camera and state names from the dataset, re-plans every second (`--n-action-steps
10` at 10 fps; LeRobot's default of 50 is five seconds open loop) and writes the
run to `outputs/train/smolvla/`. Any other flag passes through unchanged, so
`--batch_size=2` or `--resume=true` work as in LeRobot. `--policy act` trains the
control baseline the same way.

### 3. Evaluate

```bash
uv run eval.py --policy policy.py --split dev --episodes 30 --out outputs/runs/hold-dev.json
uv run eval.py --system examples/systems/lerobot.json --obs pixels --max-chunk 50 \
    --split dev --episodes 30 --out outputs/runs/smolvla-dev.json
uv run eval.py --system examples/systems/lerobot.json --obs pixels --max-chunk 50 \
    --split test --episodes 10 --out outputs/runs/smolvla-test.json
uv run -m tools.results verify outputs/runs/smolvla-dev.json
uv run -m tools.results compare outputs/runs/hold-dev.json outputs/runs/smolvla-dev.json
```

Evaluation runs seeded episodes of 240 simulated seconds, saves every state and
action, and never overwrites a result. Policy exceptions count as failures and
stay in the denominator; the summary carries a 95% Wilson interval. `verify`
recomputes every score from the saved trajectory without the policy; `compare`
pairs two runs by seed and refuses to mix observation contracts, splits or
horizons. Use a multiple of three dev episodes for equal counts per figure.

To plug in your own policy, copy `policy.py` and implement
`Policy(robot, seed).act(observation)`; return one action or a `(K, 8)` chunk with
`--max-chunk K`. `examples/systems/lerobot.json` points
`examples/lerobot_policy.py` at the run's `checkpoints/last/pretrained_model`; the wrapper predicts one
chunk per call and repeats each action 50/fps ticks, so `--max-chunk` is
`n_action_steps × 50 / fps`.

## Language models over an API

The second track asks whether a frontier language model, given the cameras and
the state, can drive the arm at all. `adapters.py` sends each observation to the
OpenAI Responses API or the Anthropic Messages API and parses a JSON chunk of
joint actions back. With `--obs pixels` both cameras go along as images.

```bash
cp .env.example .env            # put your keys and model IDs there; .env is git-ignored
uv run eval.py --system examples/systems/openai.json --obs pixels --max-chunk 25 \
    --episodes 3 --steps 3000 --out outputs/runs/openai-dev.json
uv run eval.py --system examples/systems/anthropic.json --obs pixels --max-chunk 25 \
    --episodes 3 --steps 3000 --out outputs/runs/anthropic-dev.json
```

The system files name the model through `TANGRAM_OPENAI_MODEL` and
`TANGRAM_ANTHROPIC_MODEL`, request structured JSON output, and log every request,
response and token count next to the trajectory (never the key). Keep
`--max-chunk` at 25 or more: each call is one round trip, and an episode of 3,000
control steps at chunk 25 is 120 calls. Set `chunk_size` in the system file to
match. These runs cost real money; the audit sidecar reports token usage so the
bill is predictable.

## Reference controller

`examples/oracle.py` assembles any silhouette from its solution certificate using
only the joint-action interface and physical contact. It chooses among the four
equivalent knob grasps and two carry heights by checking IK at pickup, transit and
placement, hovers above the target, descends slowly, releases 4 mm above the table
once the piece is observed within 2.5 mm and 2.5°, re-grasps a piece it loses and
re-places one that lands on a neighbour. It declares `access = "oracle"` and is a
data generator and an execution ceiling, never a leaderboard entry.

## Results

Reference controller, Panda, train seeds 0–9, 12,000 steps, 2026-09-10, sources
at commit `8f560f4` plus the controller changes logged in
[results/LOG.md](results/LOG.md):

| Silhouette | Success (10 seeds) | 95% Wilson | Final IoU of successes | Time to success |
| --- | --- | --- | --- | --- |
| square | 3/10 | 0.11–0.60 | 0.977–0.979 | 219 s |
| rectangle | 5/10 | 0.24–0.76 | 0.965–0.979 | 211 s |
| house | 3/10 | 0.11–0.60 | 0.953–0.978 | 203 s |
| cat | 4/10 | 0.17–0.69 | 0.978–0.979 | 197 s |
| **all** | **15/40** | | | |
| SmolVLA, dev / test | pending | | | |
| OpenAI / Anthropic, dev | pending | | | |

Successful episodes place every piece within about 1 mm. Failures, most frequent
first: a placed piece disturbed by a later pick or place; the slab slipping out of
the fingers during transport; IK blocked near the robot base. Every experiment,
including the negative ones, is in [results/LOG.md](results/LOG.md); summaries are
versioned in `results/` and raw trajectories stay in `outputs/runs/`.

## Reference

- [docs/protocol.md](docs/protocol.md): the contract. Units, seeds, observation
  fields, scoring rules, artifact formats, comparison rules.
- [docs/plan.md](docs/plan.md): state of the base, baseline choice with sources,
  milestones and next steps.
- `uv run view.py --result outputs/runs/x.json --episode 0` replays a saved episode
  from its recorded model and states; `uv run -m tools.history outputs/runs` builds
  a local history page next to the results.

```text
env.py                  scene, reset, state and pixel observations, CPU/GPU physics (388 lines)
tangram.py              piece geometry and exact scoring (156)
shapes.py               silhouettes, certificates, splits, prompt (91)
benchmark.py            action chunks, trajectory scoring, Wilson and paired statistics (197)
eval.py                 seeded runner, failure accounting, result artifacts (403)
adapters.py             local checkpoints, HTTP, OpenAI and Anthropic policies
policy.py               the policy interface, 19 lines, holds position
teleop.py               keyboard TCP targets through damped IK (133)
view.py                 viewer, teleoperation, recording, replay (368)
replay.py               episode reconstruction (51)
tools/prepare.py        robot assets at a pinned revision
tools/collect.py        record demonstrations for one silhouette
tools/export.py         episodes -> LeRobot v3 dataset
tools/train.py          lerobot-train with the laptop defaults and the repo paths
tools/results.py        verify and compare results offline
tools/history.py        local history page
examples/oracle.py      reference controller (338)
examples/lerobot_policy.py  LeRobot checkpoint as a pixel policy
examples/systems/       system files: hold, http, openai, anthropic, lerobot
tests/                  geometry, physics, evaluation, pixels, data, viewer
results/                versioned result summaries and the experiment log
data/, checkpoints/, outputs/  demonstrations; downloaded weights; training and evaluation runs (git-ignored)
reproduce.sh            the whole pipeline
```

Read `policy.py` → `tangram.py` → `env.py` → `eval.py`.
