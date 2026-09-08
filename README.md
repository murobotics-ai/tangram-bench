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
> **Status:** the packed start, inset pieces with grasp knobs, and manual
> teleoperation work. Only the square target exists, so the current task is
> square relocation (`square-packed-state-v3`), not varied shape assembly.
> Observations are exact state, not camera pixels. No successful manipulation
> baseline is claimed.

## Start

Install [uv](https://docs.astral.sh/uv/) and Git, then run from the repository root:

```bash
uv sync --frozen --extra warp --extra dev
uv run -m tools.prepare
uv run view.py
```

The viewer is static after reset settling. The right column shows three panels of
identical size: the goal silhouette, the top camera and the wrist camera. The two
cameras are the policy's future pixel inputs; the goal panel is a viewer aid only
and is never observed by the policy.
Keys **1 / 2 / 3** change only the main view; **Esc** closes the window. The goal
panel currently offers square only. The target appears both on the table (visible
to all cameras, and therefore to a policy) and in the goal panel (viewer only).

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

Evaluation defaults to four GPU worlds, 50 Hz control and 60 simulated seconds per
episode. GPU runs need an NVIDIA CUDA driver; first-run kernel compilation can
take several minutes. Use `--backend cpu` for a CPU run. Results are never overwritten.

Success requires silhouette IoU ≥ 0.95, overlap < 1%, and all pieces flat, resting
and still for the final 0.5 seconds. The [protocol](docs/protocol.md) defines the
API, units, seeds and exact rules. Compare the same protocol, robot, backend,
seeds and horizon; old scattered-start results are not comparable.

## Code

```text
env.py          scene, reset, observations, CPU/GPU physics
tangram.py      piece geometry and scoring
eval.py         fixed evaluation and JSON results
view.py         camera panels and keyboard input
teleop.py       manual TCP targets and damped inverse kinematics
policy.py       minimal policy interface

tools/          robot asset download and throughput measurement
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
