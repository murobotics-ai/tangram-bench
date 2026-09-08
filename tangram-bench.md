# Tangram-Bench

**A robot manipulation benchmark built on Tangram.**

## Implementation status

The piece dimensions, inset and grasp knobs below are implemented in
`square-packed-state-v3`, together with the seeded packed reset. Only the square
target currently exists, at a separate location, so the shipped task is relocation.
A varied silhouette library, pixel observations and manipulation baselines remain
pending. Literature comparisons below are supplied research notes, not claims
validated by the implementation tests.

## What it is

Tangram-Bench evaluates robot policies on one construction task: seven rigid
pieces, one work surface, one silhouette to reconstruct. Tangram is a thinly
explored task in robot learning (see [Prior work](#prior-work)), and a useful
one because success depends on reasoning as much as on control, while the
state stays small enough to measure exactly.

## Why Tangram

Most manipulation benchmarks check a per-object condition: did the gripper reach
a pose, did the object land in a bin. The reasoning involved is shallow:
identify, grasp, place. A 2026 audit of the common suites found LIBERO matched
by a 0.09B model that ignores language, and CALVIN collapsing when block poses
are randomized within the training range ([Jiang et al.,
2026](https://arxiv.org/abs/2606.04233)). Success there is cheap to fake.

In Tangram-Bench success is a *global constraint over seven interacting pieces*:

- **Goal inference.** The target is a silhouette, not a list of poses. The
  policy must decide which piece goes where before any control problem exists.
  A wrong assignment cannot be fixed by better grasping. This step is genuinely
  hard for current models: frontier VLMs reach mean IoU 0.41 placing a single
  piece and 0.23 composing two ([TangramSR,
  2026](https://arxiv.org/abs/2602.05570)), and tend to deform pieces to match
  the outline ([TangramPuzzle, 2026](https://arxiv.org/abs/2601.16520)).
- **Long-horizon dependency.** Each placed piece constrains what remains valid
  for the rest. Early mistakes compound instead of staying local.
- **Chirality.** The parallelogram cannot reach some target orientations by
  in-plane rotation; those require a 3D flip. A policy that only reasons in the
  image plane fails a predictable subset of goals.
- **Non-unique solutions.** Several assignments can tile the same silhouette.
  Scoring is by silhouette IoU and overlap, so any physically valid
  decomposition counts and hard-coding one solution earns nothing elsewhere.
- **Millimeter precision.** Success requires IoU ≥ 0.95, overlap < 1%, and
  every piece flat and resting. Small planning or control errors show up
  directly in the score.

A single grasp-and-place skill is not enough. The policy has to get a discrete
plan right, respect physical constraints over the whole episode, and execute
precisely, and the benchmark can report which of those failed.

## Prior work

Tangram is not new to robotics, but no standardized, physics-verified benchmark
with a strict success criterion exists yet.

- **MRChaos** ([ICRA 2025](https://arxiv.org/abs/2505.11818)): the closest
  work. Learns tangram assembly from silhouette prompts by self-exploration in
  PyBullet, transfers to a UR arm with a **suction** gripper; 73.6% final
  coverage in sim and 62.4% real on its hardest set. Its metric is pixel
  coverage without overlap, rest or flatness checks, chirality is not
  addressed, and the release is a method, not a reproducible evaluation.
- **Vision-based tangram assembly line** ([Qin,
  2022](https://iopscience.iop.org/article/10.1088/1742-6596/2229/1/012017/pdf)):
  classical detect-then-place engineering, no learning or benchmark.
- **TangramPuzzle** (EMNLP 2026 Findings) and **TangramSR** (2026): tangram as
  a *pure reasoning* benchmark for MLLMs, verifier-checked, no physics.
- **Adjacent assembly benchmarks**: [FurnitureBench](https://arxiv.org/abs/2305.12821)
  (real, long-horizon, contact-rich), [RAMP](https://arxiv.org/abs/2305.09644)
  (beams and pegs), [WorkBenchMark](https://arxiv.org/abs/2606.19358) (400
  LEGO Duplo tasks, planning pipeline beats a VLA at every tier),
  [PhyBlock](https://arxiv.org/abs/2506.08708) (block planning in Genesis).
  All of these enforce a global structure constraint too, so the "global vs
  per-object" contrast above holds against LIBERO/RLBench-style suites, not
  against the assembly line of work.

What Tangram-Bench adds relative to that set: exact 2D geometry that makes the
metric unambiguous, a fixed grasp handle per piece so grasping is factored out
and reasoning is what gets measured, an explicit overlap/rest/flatness gate
against false solutions, seeded procedural resets, and a planned split between
planning and execution failures.

## Arguments against, stated plainly

- **v3 does not yet test the reasoning claim.** `square-packed-state-v3` has
  one silhouette, exact state, goal corners in the observation, and no required
  flips. Moving a packed square to another pose is a pose-assignment-plus-control
  task. The reasoning story only becomes true once silhouettes vary and
  observations come from pixels.
- **Control may dominate, not reasoning.** MRChaos and the VLM tangram papers
  suggest the assignment step is learnable in 2D; the hard part with a
  parallel-jaw gripper is likely grasping and precisely placing thin flat
  pieces. The grasp handle below removes most of the grasping half of that;
  precise placement remains, and the benchmark should report it honestly.
- **Small state invites the failure modes Jiang et al. list.** A single task
  with seeded resets is easy to overfit and to solve by shortcut. Their
  diagnostics (pose randomization, ablated inputs, significance over seeds)
  should be part of any reported result.
- **No baseline yet.** Without a reliable manipulation baseline, no claim about
  difficulty is supported; see [the protocol](docs/protocol.md).

## Start state: the packed square

The shipped protocol starts every episode with the seven pieces **already
assembled as the square**, at a seeded random position and yaw on the table.
The goal is a different silhouette shown as a shadow. The robot must take the
square apart and rebuild it into that shape. This mirrors how a physical
tangram is used: it ships packed as a square and every figure starts from it.

For:

- **Canonical start.** The only variation is the square's pose and the target.
  Difficulty is then attributable to the silhouette, not to a lucky or unlucky
  scatter, while graspability still requires validation with each robot.
- **Disassembly is a real sub-problem.** Pieces start packed, with 1 mm clearance between adjacent edges.
  The policy must choose which knob to grasp first and at what yaw, then lift
  clear of the neighbours, which is contact-rich and ordered, like
  FurnitureBench-style assembly but in 2D and measurable.
- **Free reset for real robots.** Any figure can be returned to the square, so
  figure → square → figure runs without a human in the loop. That is the
  cheapest route to sim-to-real evaluation this benchmark has.
- **The inverse task comes for free.** "Return to the box" is itself a hard,
  well-defined goal, and a good curriculum step.

Against:

- **A colocated square target would be trivial.** The current target is placed
  separately, making relocation necessary. A verified silhouette library is still
  required to test varied shape assembly.
- **Control could dominate without the handle.** Interior pieces (the small
  square, the parallelogram) have no free edge at the start. The grasp handle
  below is what makes a straight top grasp out of the packed square possible.
- **Longer horizon.** Disassembly plus assembly roughly doubles the episode;
  the 60 s budget will need revisiting.
- **One start state invites memorization.** Every demo begins from the same
  tiling. Randomizing pose and the square's tiling variant (mirrored and
  rotated dissections) is the minimum to avoid a fixed disassembly script.
- **Both distributions are worth keeping.** Scattered start tests search and
  grasp under clutter; packed start tests disassembly and goal inference.
  Report them as separate protocols, never mixed.

## Grasp handle: reasoning, not dexterity

Every piece carries a **square knob on its top face, centered on the
centroid**. That knob is the only intended grasp point. The benchmark is meant
to measure whether a policy can decide *where each piece goes*, not whether it
can pinch a thin slab off a table, so grasping is made deliberately easy and
identical across pieces.

Specification. Piece geometry follows the UNL "Tans" print sheet (classic
tangram, 10 cm large triangles) printed at 200%; the implemented square side is exactly 20√2 cm. Dimensions below are nominal,
before the edge inset.

| Item | Value |
| --- | --- |
| Tangram square | 28.3 cm side (20 cm × √2) |
| Large triangles (×2) | 20 cm legs, 28.3 cm hypotenuse |
| Medium triangle | 14.1 cm legs, 20 cm hypotenuse |
| Small triangles (×2) | 10 cm legs, 14.1 cm hypotenuse |
| Small square | 10 cm side |
| Parallelogram | 14.1 cm × 10 cm sides, 45° |
| Slab thickness | 0.5 cm |
| Piece outline | sheet tile inset 0.5 mm per edge, so packed neighbours sit 1 mm apart |
| Knob footprint | 2 cm × 2 cm, square |
| Knob height | 4 cm above the top face (4.5 cm total piece height) |
| Knob position | slab centroid, top face only |
| Knob orientation | edges parallel to the piece's longest edge |

- **One grasp primitive for all seven pieces.** Top-down approach, close on
  the knob, lift. No edge grasps, no sliding, no regrasp planning. Failures
  left over are placement and reasoning failures.
- **No flips, by construction.** For now every silhouette is one that can be
  solved with in-plane rotation and translation only. The knob face is always
  up, so the intended face at the start is the face at the end;
  a piece with the knob down cannot rest flat and would fail the flatness
  check. The chirality diagnostic is deferred to a later suite with knobless
  pieces or a mirrored parallelogram.
- **The 2 cm knob fits every piece with margin.** The small triangle is the
  tightest: its centroid is 2.36 cm from the hypotenuse and 3.33 cm from each
  leg, so with the knob edges parallel to the hypotenuse 1.4 cm of slab remain
  beyond the knob, and even a corner pointed at the hypotenuse would leave
  9 mm. Every other piece keeps at least 1.9 cm in any orientation. The
  "parallel to the longest edge" rule stays as the one convention.
- **Graspability is a validation target, not yet a demonstrated result.** The closest knob
  centres are approximately 6.9 cm apart, leaving 4.9 cm free between
  neighbouring knobs. The design analysis supplied with this
  vision used the Panda finger from the
  Menagerie model (2.1 cm wide, 2.35 cm deep beyond the pad face, 8 cm
  opening): each piece has a collision-free closing axis even with the
  square mathematically closed. That check was done at half scale, where the
  worst cases kept about 70% of the yaw range; at the current scale every
  clearance doubled while the finger did not, so the margin is larger. The
  PiPER (7 cm opening) must pass the same check when its finger geometry is
  measured. The 1 mm gap therefore exists for print tolerance and contact
  stability, not for the gripper.
- **The metric does not change, the ceiling does.** Scoring projects the slab
  footprint; the knob stays inside it. Because pieces are inset, a perfect
  assembly covers 783 of the 800 cm² square, an IoU of 0.979 against the
  mathematical silhouette. The 0.95 threshold stays, which leaves roughly
  3.5 mm of whole-assembly misplacement before failure (a 4 mm global
  offset alone scores 0.945). That is the intended "millimetre precision".
- **Sim-to-real stays cheap.** The pieces are 3D-printable with the knob as
  part of the print, and the same knob is a clean fiducial for pose tracking.

## Decisions that close the piece spec

- **Slab thickness: 0.5 cm.** Total piece height 4.5 cm; the thin slab keeps
  the largest piece under 1 N so the stock grippers can lift it. The on-table
  check is "body origin within 2 mm of z = 2.5 mm".
- **Knob orientation: parallel to the longest edge.** Hypotenuse on the three
  triangle sizes, any side on the square, the 14.1 cm side on the
  parallelogram. One rule, no per-piece table.
- **Parallelogram handedness: the sheet's.** Read the sheet normally and the
  parallelogram leans right, with the top edge shifted right of the bottom
  edge; the code's tiling has the same lean. The knob goes on the printed
  face. Every silhouette in the library must be solvable with that hand and
  no flips.
- **Goal representation: a polygon.** Counter-clockwise vertices in world xy,
  optional holes, given as the `goal` observation; the square is the 4-vertex
  case. The IoU code already scores against an arbitrary polygon. A library
  entry is the polygon plus one verified solution (seven poses) used as the
  planning oracle and as proof that no flip is needed. The shadow shown in
  pixel modes is rendered from that polygon and never includes knobs.
- **Packing clearance: 0.5 mm inset per edge, 1 mm gap.** The packed reset
  places pieces at the exact tiling centroids; the inset supplies the gap.
  No exploded layout: a radial explosion leaves some neighbours still
  touching and is not needed for grasping (see above).
- **Finger rule.** Any parallel-jaw finger up to 2.4 cm deep beyond its pad
  and 2.1 cm wide can extract every piece from the packed square. Both robots still need extraction tests with their actual collision geometry;
  the dimensions alone do not prove a collision-free approach and lift.
- **Body origin stays at the slab centroid, mid-thickness.** The knob is a
  second geom on the same body. Flatness (body z axis within 5° of
  vertical) and stillness checks are unchanged; the knob raises the centre of
  mass but does not enter the metric.
- **Knob appearance.** Same colour as its piece in simulation. On printed
  pieces the 2 cm top face is reserved for an ID marker for real-world
  tracking; it is not part of the simulated observation.

## Current scope

The shipped protocol is [`square-packed-state-v3`](docs/protocol.md): seven inset
pieces with rigid top knobs begin packed as a square at a random pose, away from
the target location. The viewer shows three equal 4:3 panels: the goal silhouette,
the top camera and the wrist camera. The two cameras are what a policy will see in
the pixel protocol; the goal panel is viewer-only and never an observation. The target
also lies on the table as a visual silhouette, so the cameras do carry it. Both Panda
and PiPER models include collidable knobs with mass and inertia. Today observations
remain exact state. Lights, colours and camera poses are fixed by the benchmark and
listed in the protocol's Rendering section, because they will define the pixel
observations later.

Body origins and knobs use the **nominal tile centroid before inset**, at slab
mid-thickness. This preserves exact tiling placements; the inset triangle's actual
area centroid can shift slightly. MuJoCo computes the combined center of mass
from the slab and knob; it is distinct from the body origin.

The metric projects the inset slab only. The table-height reference is 2.5 mm;
flatness now explicitly requires the knob face up. Thresholds remain IoU ≥ 0.95,
overlap < 1%, and the final 0.5-second hold. Versions with different piece geometry
or reset distributions are not comparable.

Only square is implemented. Because source and target occupy different locations,
this task is not solved at reset. No policy was run to validate this geometry update.
The existing `examples/scripted.py` is an old controller attempt, not a baseline
validated for knobs. Physical material parameters (700 kg/m³ and friction) remain
assumptions. Geometric fit is not proof of reachability or reliable extraction.

Next: add target polygons with verified seven-piece solutions, test approach and
extraction clearance on both robots, establish a manipulation baseline, then add
pixel observations and real-world validation. Polygon holes and alternative
packing variants are not implemented. Keep policy experiments under the fixed
[protocol](docs/protocol.md); workflow instructions are in [README.md](README.md).

## Manual validation tooling

`uv run view.py --teleop` enables keyboard control of TCP translation/orientation
and gripper opening, with simultaneous context, top and wrist views. Small damped
inverse-kinematics updates drive the same bounded actuators as the environment.
There are no object attachments or direct piece pose commands. Pause, fine movement
and same-seed reset support manual inspection. No policy, scoring loop or recording
runs here; demonstration collection remains deferred until manipulation is validated.
The controller uses shortest-rotation orientation error, adaptive damped least
squares, bounded steps with residual checks, and a reset-posture preference in
Panda's redundant subspace. It reports IK residual separately from physical tracking
in both position and orientation. This is local IK, not collision-aware planning.

Validation on 2026-09-08: the previous and revised controllers both met 1 mm /
1 degree IK tolerances on 61 nearby/limit-case reachable targets per robot; neither
accepted the far-away target. Revised median solve time was about 1 ms on this host.
A single upward-motion-and-settle CPU check left 4.7 mm / 0.63 degrees tracking error
on Panda and 0.5 mm / 0.13 degrees on PiPER despite negligible IK residual. These
are limited controller diagnostics, not contact or grasp validation; Panda's physical
tracking still needs investigation before claiming millimetre placement accuracy.
Controls are documented in [README.md](README.md).

## Roadmap

Beyond the square, Tangram-Bench plans separate planning, execution and
end-to-end modes, plus diagnostic suites (chirality, recovery) so failures can
be attributed to reasoning or control rather than collapsed into one number.
The same construction-through-action idea could later extend to other domains
under the Mu-Bench name (3D construction, packing, kitting). None of that is
implemented yet.
