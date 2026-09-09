<p align="center">
  <img src="assets/preview.png" alt="Packed tangram square, a Franka Panda and the target shadow on the table" width="760">
</p>

<h1 align="center">Tangram-Bench</h1>

<p align="center"><b>Seven pieces, one arm, one target silhouette.</b></p>

<p align="center">
Start with the pieces packed as a square at a random pose. Rebuild the shape shown by
the shadow. Success is a global constraint over all seven pieces, scored exactly.
</p>

<p align="center">
  <a href="https://github.com/murobotics-ai/tangram-bench/actions/workflows/test.yml"><img src="https://github.com/murobotics-ai/tangram-bench/actions/workflows/test.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/status-prototype-orange" alt="status: prototype">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="python 3.12+">
  <img src="https://img.shields.io/badge/physics-MuJoCo%203.6-2c3e50" alt="MuJoCo 3.6">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

<p align="center">
  <a href="docs/protocol.md"><b>Protocol</b></a> ·
  <a href="tangram-bench.md">Vision and rationale</a> ·
  <a href="#teleoperate">Teleoperation</a> ·
  <a href="#plug-in-a-policy">Plug in a policy</a>
</p>

MuJoCo physics, optional GPU batching with MuJoCo Warp, Franka Panda or AgileX
PiPER. Plain Python; no training framework.

> [!NOTE]
> **Public pilot:** square, rectangle, house and cat silhouettes have verified
> seven-piece solutions without flips. Policies receive exact state and an optional
> text prompt. The physical reference controller remains experimental; geometric
> validation does not establish reliable robot assembly or sim-to-real transfer.

## Start

Install [uv](https://docs.astral.sh/uv/) and Git, then run from the repository root:

```bash
uv sync --frozen --extra dev
uv run -m tools.prepare
uv run view.py
```

The rectangular top panel displays the exact evaluation prompt. Policies receive
it as `observation["prompt"]` and may ignore it. The right column shows the goal,
top camera and wrist camera. These camera panels are inspection views; this
protocol gives policies state, not pixels. Keys **1 / 2 / 3** switch the main camera.

```bash
uv run view.py --target house
uv run view.py --target cat --prompt "Assemble the tangram to match the silhouette."
```

```bash
uv run view.py --robot piper --camera wrist --seed 42
MUJOCO_GL=egl uv run view.py --camera top --out runs/scene.png
```

## Teleoperate

```bash
uv run view.py --teleop --robot panda
uv run view.py --teleop --robot piper
uv run view.py --teleop --backend warp    # optional CUDA physics
```

The keyboard stands in for a 3Dconnexion SpaceMouse, the usual Panda teleoperation
device: six held axes for the puck plus its two buttons. The axes follow Isaac Lab's
`Se3Keyboard`; the gripper and reset sit on R, F and O so the whole session runs
from the left hand. Every key does exactly one thing; all motion is world-frame.

- **W / S:** +X / −X; **A / D:** +Y / −Y; **Q / E:** up / down (tilt and push the puck).
- **Z / X:** roll; **T / G:** pitch; **C / V:** yaw (twist the puck).
- **R / F:** open / close the gripper while held (left button). Release to stop the
  jaws at any opening.
- **O:** reset the same seed (right button).
- **Shift:** fine movement; **P:** pause physics.
- **1 / 2 / 3:** main camera; **Esc:** exit.

The three views update together. Translation is 8 cm/s and rotation 0.6 rad/s;
Shift reduces both to one fifth. The panel separates **IK residual** (returned joint target versus requested TCP)
from **physical tracking error**, both in millimetres and degrees, and shows commanded
gripper opening. IK reports OK below 1 mm / 1 degree; NOT CONVERGED means the local
solver did not find a solution within its budget, not proof that none exists. Joint limits and the existing actuator slew limiter remain active.
The target stays within 4 cm and 0.35 radians of the actual TCP to limit accumulation
against obstacles. Damping increases near singularities; worsening IK steps are
rejected. Panda also softly prefers its reset joint posture in the redundant
subspace, helping control elbow posture without prescribing a Cartesian elbow pose.
IK is local, not collision-aware motion planning: contacts and reach limits can prevent
tracking. Use the error display, fine motion and reset while testing grasps.

Single-world teleoperation defaults to CPU physics with OpenGL rendering; `--backend warp` uses the existing GPU backend. Control runs at 50 Hz simulated time, separately
from drawing; slow machines run slower rather than taking large physics steps.
Unfocused windows ignore movement keys. Pause freezes physics, including held pieces.
No policies, success evaluation or recording run in this mode. Grasp reliability is
still to be validated; teleoperation provides the tool to do that.

## Plug in a policy

Copy `policy.py` and implement `Policy(robot, seed).act(observation)`. Return arm
joint targets in radians followed by gripper opening in [0, 1]. The included
policy holds position; it is a negative control.

```bash
uv run eval.py --policy my_policy.py --out runs/my-policy.json
```

Evaluation defaults to four CPU worlds, 50 Hz control and 60 simulated seconds per
episode. Four episodes are a smoke test. For GPU batching, install the `warp` extra
and select `--backend warp`; this needs an NVIDIA CUDA driver. Results and their
artifact directories are never overwritten.

Each run saves a versioned JSON result, a compiled model, an episode journal, and
compressed trajectories containing every observation, requested action and applied
actuator command. Policy exceptions and invalid actions count as failed attempts;
physics or setup failures invalidate the run. By default `--target suite` selects
figures from the requested split: train = square/rectangle, dev = house, test = cat.
These are public figure holdouts, not secret test data. `--target NAME` selects a
single diagnostic figure. The summary includes a 95% Wilson
interval and keeps policy failures in the success-rate denominator.

```bash
# Recompute scores without loading the policy.
uv run -m tools.results verify runs/my-policy.json

# Development comparison: identical settings and seeds for both candidates.
uv run eval.py --policy policy.py --episodes 100 --out runs/hold-dev.json
uv run eval.py --policy my_policy.py --episodes 100 --out runs/candidate-dev.json
uv run -m tools.results compare runs/hold-dev.json runs/candidate-dev.json
```

Comparison requires matching evaluation settings and seeds, and reports the paired
success-rate difference with an uncertainty interval. Choose the sample size in
advance; reserve `--split test` for the frozen final candidate. The
[protocol](docs/protocol.md) specifies compatibility checks and reporting rules.

Policies may return a `(K, action_dim)` chunk with `--max-chunk K`. The runner
validates the entire chunk and executes it open loop, one action per control tick.
Use `--policy-metadata metadata.json` to record training data, compute and inference
configuration, and repeated `--policy-artifact path` options to hash checkpoints
and imported code. Policies must seed their own RNGs from the constructor seed.

Success requires silhouette IoU ≥ 0.95, overlap < 1%, and all pieces flat, resting
and still for the final 0.5 seconds. The [protocol](docs/protocol.md) defines the
API, units, seeds and exact rules. Compare the same protocol, robot, backend,
seeds and horizon. v5 adds the silhouette corpus and prompt contract; results from
older protocols are not comparable.

## Systems and checkpoints

`--system` loads an adapter configuration instead of a Python policy entrypoint:

```bash
uv run eval.py --system examples/systems/hold.json --out runs/hold.json
uv run eval.py --system examples/systems/http.json --max-chunk 25 --out runs/server.json
```

A local system specifies `module`, optional `checkpoint`, and constructor `kwargs`.
Paths are relative to the JSON file; the checkpoint path is passed to the policy's
constructor and its contents are hashed. The policy supplies the architecture and
normalization required to load those weights.

The OpenAI and Anthropic examples use separate API adapters. Set `OPENAI_API_KEY`
and `TANGRAM_OPENAI_MODEL`, or `ANTHROPIC_API_KEY` and `TANGRAM_ANTHROPIC_MODEL`, then:

```bash
uv run eval.py --system examples/systems/openai.json --max-chunk 25 --out runs/openai.json
uv run eval.py --system examples/systems/anthropic.json --max-chunk 25 --out runs/anthropic.json
```

Choose an exact model ID supported by your account. These adapters send state and
the prompt, parse joint-action JSON, and record requests, responses and token usage.
`use_prompt: false` omits the instruction for a remote policy. HTTP adapters accept
`{"actions": [[joint_targets..., gripper_opening], ...]}`. Local policy code can
perform additional observation/action conversions. Inference calls are bounded by
`--max-inference-calls`; provider output is capped by `--max-output-tokens` per call.
Remote requests have a timeout. Provider contracts are covered by mocked tests;
authenticated provider runs are not included in this repository.

## Inspect outputs and checkpoint history

```bash
uv run view.py --result runs/hold.json --episode 0
uv run -m tools.history runs --out runs/history.html
```

Replay uses the saved model and states: **Space** pauses, **Left/Right** seek one
second, and **Home/End** select the first/last state. `--frame N --out image.png`
exports a replay frame. It never reloads or executes the policy.

Open `runs/history.html` in a browser. It groups compatible runs, plots success
against evaluation date, shows checkpoint hashes, and includes a top-down episode
player with the exact prompt. The HTML is self-contained and can be regenerated
from results. Older source versions remain visible as unverified historical records;
they are not mixed into verified comparisons. Provider audit files live beside the
trajectories. This is a local viewer, not a hosted submission service.

```bash
uv run -m shapes  # verify every silhouette certificate
uv run eval.py --policy examples/oracle.py --target square --steps 12000 \
  --max-inference-calls 12000 --out runs/oracle.json
```

The oracle uses solution certificates and moves pieces through physical contact.
It is labeled separately in comparisons and is for diagnosing manipulation; it
has not achieved a validated full-assembly success rate on both robots.

## Run a working house demonstration

`examples/house.py` physically assembles the house with Panda on development seed
100076. The checked demonstration reaches approximately 0.976 final IoU with all
seven pieces flat and still, zero overlap, and success held through the horizon.
It chooses among equivalent knob grasps, checks IK at pickup and placement,
transits away from the base, and corrects placement using the observed piece pose.
All movement uses the standard joint-action interface and physical contacts.

From the repository root:

```bash
uv run eval.py --policy examples/house.py --robot panda --target house \
  --split dev --offset 76 --episodes 1 --num-envs 1 \
  --steps 12000 --max-inference-calls 12000 --policy-artifact teleop.py \
  --out runs/house-demo.json
uv run -m tools.results verify runs/house-demo.json
uv run view.py --result runs/house-demo.json --episode 0
```

Evaluation finishes before replay opens. Space pauses replay, Left/Right seek,
and 1/2/3 switch the main camera. Use a new output filename for another run.

This is a **selected-scene demonstration**, not an estimated benchmark success
rate: it uses the published solution certificate (`access = "oracle"`), one
development seed, and 240 simulated seconds rather than the default 60 seconds.
Other seeds, PiPER, and completion within the standard horizon are not validated
by this demonstration. `tests/test_house.py` exercises the complete physical run.

## Code

```text
env.py          scene, reset, observations, CPU/GPU physics
tangram.py      piece geometry and scoring
shapes.py       silhouettes, certificates, splits and default prompt
adapters.py     local/checkpoint, HTTP and provider policy adapters
benchmark.py    policy chunks, trajectory scoring and paired statistics
eval.py         seeded runner, failure accounting and run artifacts
view.py         prompt, camera panels, teleoperation and replay
replay.py       reconstruction from recorded models and states
teleop.py       manual TCP targets and damped inverse kinematics
policy.py       minimal policy interface

tools/          asset download, throughput, offline verification and comparison
examples/       experimental controller, not a validated baseline
tests/          geometry, physics and viewer checks
```

Read `policy.py` → `tangram.py` → `env.py` → `eval.py`.

## Develop

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
MU_BENCH_TEST_WARP=1 uv run pytest -q    # optional GPU checks
uv run -m tools.bench --out runs/perf.json
```

Keep functions small and dependencies few. For policy experiments, keep the
environment and evaluator fixed; change one candidate, compare identical dev
seeds, and record data/checkpoints/compute. Reserve test seeds for reporting.
Protocol changes need a new version. Timing alone is not evidence of correctness.

Code is [MIT licensed](LICENSE). Downloaded Menagerie assets retain their upstream
licenses. Training and checkpoints belong to each candidate policy.
