# square-packed-state-v3

An inspectable prototype protocol, not a claim of general Tangram competence.
The benchmark implementation lives in `tangram.py`, `env.py`, and `eval.py`.

## Episode

Seven free rigid convex prisms start assembled as a square, with a shared seeded
translation and yaw. The robot is mounted on a table at z = 0. Source-square center
is uniform in x ∈ [0.28, 0.36], y ∈ [-0.30, -0.24] metres; yaw is uniform in [-π, π].
The fixed dissection is transformed rigidly, without individual piece randomization.
Each nominal edge is inset by 0.5 mm, leaving 1 mm between neighboring tiles.
Origins stay at nominal tiling centroids; there is no radial expansion.

An independent random stream places the target square near (0.32, 0.20), on the +y
side and nearer the base than the source, with ±0.015 m translation and uniform yaw.
Source and target are disjoint by at least 4.8 cm over the first 300 seeds; the
farthest piece corner from the base is 0.66 m. The current task is
square relocation; arbitrary silhouettes and shape generalization are not implemented.
The viewer displays the target silhouette both on the table and in a separate goal panel.
The table silhouette is visual only: it adds no contacts or mass. The goal panel is a
viewer aid and is not part of any observation; a policy sees the target only as the
`goal` corners (state protocol) or as the silhouette on the table through the top and
wrist cameras (future pixel protocol). The goal, top and wrist panels share one 4:3
viewport size, beside the main context view. The top camera sits 1.3 m above the
table at (0.30, 0) with a 44° vertical field of view, so its frame covers the whole
table and never the void beyond it. The wrist camera is fixed to the last arm link on
the face that points up at the home pose, behind the fingers, and looks along the
tool axis so the fingertips stay in the lower part of its frame (Panda: 75° field of
view; PiPER: 60°, which keeps the wider gripper body out of frame).

Geometry: 0.20√2 m nominal square (the UNL tangram sheet at 200%), 0.005 m slabs,
0.02 × 0.02 × 0.04 m top knobs.
Each knob shares its piece body, color, density (700 kg/m³) and friction, and is
aligned with the longest slab edge. Combined mass/inertia include the knob; scoring
uses only the inset slab. Body origins and knob positions use nominal pre-inset
centroids, while MuJoCo computes the physical center of mass separately.
Packed-start graspability and manipulation remain unvalidated.
Both backends begin after the same 100 CPU settling steps. The table is a solid box
to avoid the mesh/plane penetration observed in early checks.

This protocol replaces `square-state-v0` scattered resets, packed v1 plain prisms and
packed v2 at half scale with the target at (0.48, 0). None of those results are comparable. Observation fields and thresholds
remain unchanged; footprints now include the inset and flatness requires face up.

Defaults: 500 Hz physics, 50 Hz control, 3,000 actions (60 simulated seconds).
The full horizon always runs; first sustained success is also recorded.
One policy instance per episode; a fresh instance resets its internal memory.
Wall time is diagnostic, not a normalized compute budget.

## Rendering

Every visual parameter is set by the benchmark, not by the robot file, so both
robots render under the same light. These values will define the pixel observations
of a future protocol; changing any of them changes that protocol's version.

| Item | Value |
| --- | --- |
| Key light | point light at (0.30, -0.40, 1.50) m, direction (0, 0.2, -1), diffuse 0.70, specular 0.10, ambient 0.05, casts shadows |
| Headlight (camera-attached) | ambient 0.15, diffuse 0.35, specular 0.10 |
| Robot file lights | removed at load; Menagerie's `top` (Panda) and `spotlight` (PiPER) do not exist in the scene |
| Table | 2 × 2 × 0.1 m box, top at z = 0, colour (0.89, 0.88, 0.85) |
| Background | MuJoCo default gradient, black beyond the table |
| Piece colours | orange, sky blue, red, yellow, brown, blue, green, in piece order (two large triangles, medium, two small, square, parallelogram); knobs share their piece colour |
| Target shadow | flat box 0.2 mm thick, colour (0.12, 0.14, 0.16), visual only |
| Top camera | (0.30, 0, 1.30) m looking straight down, 44° vertical field of view |
| Wrist camera | Panda: (0.065, 0, 0.04) in the hand frame, 75°; PiPER: (-0.09, 0, 0.06) in link6, 60°; both look along the tool axis |
| Context camera | (1.10, -1.35, 1.05) m aimed at (0.28, -0.10, 0.23), 50°; inspection only, never an observation |
| Offscreen resolution | 1440 × 960 for the viewer frame; panels are 4:3 crops |

The lighting was reduced from the earlier setup (two 0.7 diffuse lights plus the
default 0.4 headlight on a 0.92 table), which saturated about 4.5% of the context
view at a mean grey level of 100. The current values give a mean of 82 and saturate
under 1%.

## Observations

All arrays are independent copies. Frame: robot base/world; table top z = 0.

- `robot`: `panda` or `piper`.
- `time`: elapsed benchmark seconds, excluding reset settling.
- `qpos`, `qvel`: arm joints followed by two finger joints; length 9 for Panda, 8 for PiPER.
  Revolute joints use radians; finger slides use metres.
- `tcp_pos`: (3,) world tool reference point in metres.
- `tcp_mat`: (3, 3) tool-to-world rotation.
- `pieces`: (7, 7) body-origin xyz and unit wxyz quaternion. Origins are nominal pre-inset tile centroids.
- `piece_velocities`: (7, 6) free-joint velocity: translational xyz in world coordinates,
  then angular velocity in the local frame, following MuJoCo's convention.
- `goal`: (4, 2) ordered square corners in world xy, not target piece poses.

Piece order: large triangle, large triangle, medium triangle, small triangle,
small triangle, square, parallelogram. Local polygon vertices are public constants
in `tangram.py`. The current task does not require chirality-changing flips.

## Actions and policy boundary

Shape (8,) for Panda, (7,) for PiPER: absolute arm joint position targets in radians,
then normalized opening (0 closed, 1 open). Invalid shapes, nonfinite values,
limits violations and out-of-range gripper values raise an error rather than earn
a score. There is no direct object action, attachment shortcut, or pose assignment.
Targets pass through a fixed 0.04 rad/action slew limiter and the robot's position
actuators. The robot models differ in dynamics, so results must be kept separate.

Gripper actuators are stiffened tenfold over the Menagerie files, because a position
servo squeezes a 2 cm knob with only stiffness × 1 cm of gap: Panda tendon gain
1000 N/m and damping 100 (about 10 N on a knob), PiPER finger kp 400, kv 20, force
limit 40 N (about 4 N). Command ranges are unchanged. With 5 mm slabs the heaviest
piece weighs 0.78 N.

The example policy can build its own kinematic model, but is not given the live
evaluator's simulator. This is a cooperative API boundary, not process isolation.
External checkpoints, imports and training provenance must be documented by the
submitter; the runner hashes the main policy file but cannot capture every external dependency.

## Metric

Project each actual piece polygon into world xy using its full body quaternion.
Let U be their union and G the target polygon. IoU = area(U ∩ G) / area(U ∪ G).
Overlap fraction = (sum of piece footprint areas − area(U)) / area(G).

A timestep passes if:

- IoU ≥ 0.95 and overlap fraction < 0.01;
- every body origin is within 2 mm of half-thickness above the table;
- every local z axis is within 5 degrees of the upward world vertical direction;
- each piece's linear speed is < 0.01 m/s and angular speed < 0.1 rad/s.

Episode success means the final 25 consecutive timesteps pass. Report mean success
and mean final IoU over the same seeds. `first_success_seconds` records the first
25-step hold, even if the policy later disturbs the solution. We do not
require the gripper to be empty at completion; adding that would change the protocol.
Tests explicitly reject airborne, moving, tilted and overlapping false solutions.

## Splits and reporting

Train seeds: [0, 100000); dev: [100000, 200000); test: [200000, 300000).
The runner selects a consecutive range via `--offset` and `--episodes`. All splits
share the same shape distribution. Public seeds provide repeatability, not secrecy.
Do not tune on test or claim unseen-shape generalization from these partitions.

Compare policies at the same robot, physics backend, dependencies, episode seeds,
observations, control rate and horizon. Keep evaluator files fixed. Record checkpoint
hashes, training data, compute, external model calls and any privileged information
alongside the runner JSON. Tiny smoke tests are not leaderboard evidence.

Both backends use the same compiled MJCF. Numerical/contact trajectories need not
be bit-identical; the test suite checks a short resting scene, not manipulation
parity. Warp batches physics while Python policy calls and scoring remain serial.

PiPER capsule/mesh pairs have limited multicontact support in Warp. Reset agreement
does not establish contact-rich manipulation parity or successful knob extraction.
