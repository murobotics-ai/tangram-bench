# Experiment log

One entry per experiment, newest last, negative results included. Summaries of
the runs live next to this file as dated JSON; raw trajectories stay in `outputs/runs/`
(not committed). Every number in the README points here.

## 2026-09-10: reference controller across seeds (baseline 2/20 -> 14/40)

Setting: Panda, train seeds 0-9 per figure, 12,000 steps, default physics.
File: `2026-09-10-oracle-panda.json`.

| Figure | Success | 95% Wilson |
| --- | --- | --- |
| square | 3/10 | 0.11–0.60 |
| rectangle | 5/10 | 0.24–0.76 |
| house (pentagon, since replaced) | 2/10 | 0.06–0.51 |
| cat | 4/10 | 0.17–0.69 |
| all | 14/40 | |

What changed from the pilot controller (`examples/house.py`, one validated seed):
generalized to every figure; grasp chosen over two carry heights (0.22 m and
0.14 m, the low one for poses near the base where the elbow hits its limit);
hover 3 cm above the target before a 0.6 mm/tick descent; release 4 mm above
the table instead of 8 mm (drops from 8 mm landed on neighbours); anti-windup
on the servo bias; a speed ramp after each transition; re-grasp when a piece
is lost, re-descend when it touches down off target, re-place when it rests on
a neighbour.

Failure modes in the final run, most frequent first: a placed piece disturbed
by a later approach, descent or retreat (15 of 26 failures); slab creeping
out of the fingers during transport (4, plus 1 still resting on a neighbour
after three re-placements); IK blocked near the base (6).

Negative results:
- `noslip_iterations=5` removes the finger creep almost entirely (grasp offset
  20 -> 20.5 mm instead of 20 -> 33 mm) but MuJoCo Warp raises
  NotImplementedError for the noslip solver. Not adopted.
- `impratio=10` also removes the creep, but the slab then pivots about the
  finger contact during descent and lands off target on every probed seed.
  Not adopted.
- Carry height 0.10 m everywhere: 0/10 on square; the short descent pushed
  pieces into the table before planar alignment converged.
- Faster carry (2.4 mm/tick): 0/10 on square, more slips.

## 2026-09-10: SmolVLA smoke fine-tune on the laptop (feasibility only)

`lerobot-train` on `lerobot/smolvla_base`, 8-frame smoke dataset from the hold
policy, batch 4, 40 steps on an RTX 5060 Laptop (8 GB): peak 2.4 GB VRAM,
0.12 s per update. Inference through `examples/lerobot_policy.py`: 14 s to
load, 0.16 s per 10-action chunk, 920 MiB peak. No performance claim; the
dataset was not a demonstration.

## 2026-09-10: classic house replaces the pentagon house

The first "house" was a pentagon (4x1 body plus a triangular roof). It is now the
classic tangram house: body 2√2 x √2, overhanging roof, chimney; mirrored so
the parallelogram needs no flip. This needed eighth-turn rotations in the
certificate format (the chimney square is turned 45° relative to the roof
triangles), so `shapes.py` now stores every certificate as eighth turns plus
`a + b*sqrt(2)` vertex positions. Same controller, Panda, train seeds 0–9,
12,000 steps: **3/10** (Wilson 0.11–0.60), successes at IoU 0.953–0.978, 203 s
to success. All seven failures completed off target (IoU 0.59–0.95): a placed
piece disturbed later, as on the other figures. Summary in
`2026-09-10-oracle-panda.json` (house entry replaced). Total across figures
becomes 15/40.

## 2026-09-11: smooth, self-aware reference controller (16/40 -> 20/40)

Setting: Panda, train seeds 0-9 per figure, 12,000 steps, default physics,
recorded with `tools/collect.py --workers 10 --keep-failures` (episodes stop one
second after a held success). File: `2026-09-11-oracle-panda.json`. Metrics
computed from the recorded episodes: success; episodes with the hand tilted
more than 60° from vertical at any frame; episodes with the TCP above 0.5 m
after the first 30 s; re-grasps (subtask "slipped"); the mean over episodes of
the peak TCP speed and acceleration at 10 fps.

| Revision | Success | Tilted hand | TCP > 0.5 m | Re-grasps | Peak d(speed)/dt |
| --- | --- | --- | --- | --- | --- |
| baseline (10 Sep controller) | 16/40 | 10 | 6 | 33 | 10 m/s² |
| + glide, pinned-joint IK, wound-up recovery | 14/40 | 10 | 0 | 32 | 1.5 m/s² |
| + yaw-only carry orientation | 15/40 | 5 | 0 | 32 | 1.1 m/s² |
| + IK inside soft limits, diagonal lift near base | 15/40 | 4 | 0 | 33 | 1.1 m/s² |
| + self-collision cost, posture steering | 21/40 | 4 | 0 | 19 | 1.6 m/s² |
| faster carry 0.12 m/s, free 0.175 m/s (rejected) | 6/40 | 4 | 1 | 81 | 1.9 m/s² |
| + fast grasp choice: pruning, IK early exit | 19/40 | 3 | 0 | 16 | 1.5 m/s² |
| + stall re-planning, repair pass, preallocated recorder (**final**) | **20/40** | 3 | 0 | 16 | 1.6 m/s² |

| Figure | Success | 95% Wilson | Final IoU of successes | Median time to success |
| --- | --- | --- | --- | --- |
| square | 5/10 | 0.24–0.76 | 0.962–0.979 | 211 s |
| rectangle | 5/10 | 0.24–0.76 | 0.977–0.979 | 219 s |
| house | 4/10 | 0.17–0.69 | 0.965–0.979 | 223 s |
| cat | 6/10 | 0.31–0.83 | 0.956–0.979 | 219 s |
| all | 20/40 | | | |

Grasp choice used to take 7 s of wall time per piece (7,500 damped-IK solves
plus the collision checks), during which the worker's simulation stood still
and the arm looked frozen in the `--watch` window. Two changes bring it under
0.5 s: branch-and-bound over the grasp options (exact: costs only grow along a
path, so an option already above the best complete one cannot win) and
stopping the IK refinement once converged below 1e-5. The early exit changes
the null-space postures slightly, so trajectories differ from the 21/40 run.
Physics is deterministic per seed, so 21 -> 19 is a behaviour change, not run
noise: paired by figure, rectangle 8 -> 5, square 4 -> 5, house and cat
unchanged. With ten seeds per figure the difference sits well inside the
Wilson intervals, which cannot establish equivalence either; the trade was
made for the wall-clock gain and is recorded as such. A figure now records in about 67 s with ten workers instead of 90.

Arms standing still, seen in the `--watch` window with ten workers, had three
causes, measured rather than guessed:
- Memory, not the controller: a worker appended frames to lists and stacked
  them at save time, 1.5 GB -> 2.6 GB peak per worker, and ten episodes end
  within seconds of each other, so the machine (30 GB) went into swap and
  every arm froze while one finished. The recorder now preallocates its
  arrays for the horizon: 1.5 GB peak, 11 s to write instead of 23 s.
- The 1,000-tick hard timeouts: 20 s of a parked arm before "Blocked". A
  phase that has not brought the hand a millimetre closer in 3 s (6 s with a
  piece held, since hover and lower converge slowly by design) now lets go
  and re-plans with another grasp; a stalled retreat hands over to the next
  grasp's joint-space glide. No "Blocked" error remains.
- "All seven pieces are placed; hold still" while the assembly was not
  solved: up to 78 s of holding. After the first pass the controller re-places
  the piece farthest from its certificate pose, up to three repairs.
Metric: the longest interval per episode with the TCP under 3 mm/s. Before:
mean 1.4–11 s, max 78 s. After: mean 1.2–1.4 s, max 4.6 s, on all 40 seeds.
Grasp selection itself took 7 s of wall time per piece (the worker's
simulation stands still meanwhile); it is under 0.5 s now, see above.

Correction (audit, 11 Sep): the "peak accel" column above differentiates the
TCP *speed*, so it misses acceleration from changes of direction. With the
velocity vector (10 fps finite differences) the final controller's peak TCP
acceleration on the house is 3.0 m/s² on average and 4.6 m/s² at most, peak
speed 0.33 m/s. The column is kept as recorded for the comparison across
revisions; it is not an absolute acceleration.

Root causes found by replaying failing seeds in process (probe scripts in the
session scratchpad, not the repo):
- The "arm pointing at the ceiling" pose was not a joint-limit problem at
  first. The pinch on the knob lets the slab pivot about the finger axis, and
  the carry phase re-measured the grasp transform every tick and tilted the
  hand to "correct" the piece's tilt, chasing the pivot until the wrist wound
  up against its stops (joints 6 and 7 at 3.75 and 2.90 rad) and the damped IK
  walked the TCP upward, with the 3 cm setpoint leash following it. Fix: the
  grasp correction is yaw-only; the hand stays vertical and the slab flattens
  when it meets the table.
- Blocks near the robot base (approach, descend, close) were self-collision:
  the folded forearm (link5) pressing on the shoulder column (link1), which
  the IK cannot see, leaving the TCP 7–25 mm short with the bias integrator
  saturated. Fix: `choose_grasp` runs the collision check on every path point
  (corners and midpoints) and penalizes arm-link contacts, and the null-space
  posture always steers to the configuration it checked.
- MuJoCo joint limits are soft: a command exactly on a stop settles a few
  millimetres short. The IK now solves 0.03 rad inside the actuator range.
- A step command into the grasp configuration jerked the arm (peak 10 m/s²);
  it is now a 4 s smoothstep glide in joint space.
- Safety net: a carried piece with the TCP more than 8 cm above the carry
  height or the hand tilted more than 60° triggers an immediate release,
  retreat and re-grasp with another of the knob's four grasps (the failed one
  is banned for that piece), instead of hanging for the 1,000-tick timeout.

Negative result: carrying at 0.12 m/s instead of 0.06 m/s (and free motion at
0.175 m/s) drops success to 6/40 with 81 re-grasps; the slab swings in the
pinch and lands on neighbours. Successful episodes take 219–225 s of the 240 s
horizon, so any lost time is a failure; a longer horizon would be a protocol
change and is left for a later revision of the protocol.

Remaining failures, most frequent first: the last piece (blue square) not
placed before the horizon; a piece landing on a neighbour's edge after three
re-placements; a placed piece disturbed by a later placement.

## 2026-09-11: success tolerance set to 5 mm per piece (20/40 -> 25/40)

The success test used IoU ≥ 0.95, overlap < 1%, 2 mm to the table and 5° of
tilt, which on certificate layouts corresponds to every piece within 1–2 mm.
The benchmark's question is whether a policy infers an unseen figure and the
order of its pieces; servo precision at the millimetre is a separate skill
that a 10 fps chunked policy cannot be expected to have. The thresholds are
now calibrated for 5 mm per piece (constants in `tangram.py`, method in
`docs/protocol.md`): IoU ≥ 0.87, overlap ≤ 6%, every piece at least 85%
inside the silhouette, 4 mm to the table, 8° of tilt, still for half a
second. Every piece 5 mm off passes in all but 1 of 10,000 sampled layouts;
10 mm on every piece fails in most samples; a large triangle 3 cm out fails on
the per-piece gate, which the IoU alone would let through. The whole assembly
rigidly shifted 10 mm still passes on three figures (IoU 0.88–0.90): the test
constrains the assembly more than its position, by design.

Same controller, same seeds, re-recorded (episodes stop once the looser test
holds, so a few end earlier). File: `2026-09-11-oracle-panda.json`.

| Figure | Success | 95% Wilson | Final IoU of successes | Median time to success |
| --- | --- | --- | --- | --- |
| square | 7/10 | 0.40–0.89 | 0.934–0.979 | 212 s |
| rectangle | 7/10 | 0.40–0.89 | 0.948–0.979 | 218 s |
| house | 5/10 | 0.24–0.76 | 0.958–0.979 | 218 s |
| cat | 6/10 | 0.31–0.83 | 0.956–0.979 | 219 s |
| all | 25/40 | | | |

The five episodes that flipped had the figure assembled with one slab resting
2–3 mm up on a neighbour's edge, or every piece within a few millimetres
(IoU 0.93–0.95). No episode in the 0.87–0.93 band exists: the controller
either assembles the figure or leaves a piece far away. Smoothness metrics
are unchanged (peak change of speed 1.5–1.8 m/s², about 3 m/s² as a vector
acceleration; longest still interval 4.6 s).
Numbers from the earlier entries of today were scored with the old test and
are not comparable.

## 2026-09-11: audit fixes; success 28/40 both sustained and at the horizon

An independent audit of the day's work (its report is not in the repository)
reproduced the earlier 20/40 and found eleven problems. All were confirmed and
fixed, with tests:

- Recorder memory doubled between episodes (the previous episode's arrays
  stayed referenced while the next allocated): each episode now records inside
  its own function scope; a weakref test checks one recorder alive at a time.
- Teleoperation recording could not save after the preallocation change
  (`Session.finish` tested an array's truth value) and had a fixed capacity:
  it uses the frame count and the recorder grows when a session runs long.
- A worker killed without reporting (out of memory, `os._exit`) hung the
  collection, and worker errors still gave exit code 0: the parent watches
  exit codes, lists the seeds it lost, and the CLI exits 1. A failing
  `--watch` window no longer aborts the recording. Episode files are written
  to a temporary name and renamed.
- The controller repaired assemblies the benchmark already accepted (the
  repair pass compared to the certificate), and, worse, after releasing the
  last piece it re-grasped a slab resting 2–4 mm up because its "settled" test
  used the old 2 mm tolerance: two of 25 sustained successes were destroyed
  before the horizon. Both now defer to the success test; an accepted
  assembly is never touched again.
- The stall detector ignored orientation progress (a hand turning in place
  was "stalled"); the repair pass ignored height and tilt; the collision cost
  skipped hand and fingers (name-prefix filter). All three fixed.
- Metrics: the summary script derived acceleration from speed, used an
  off-by-one median and estimated the first success from the stop time. It is
  now `tools/report.py`, versioned and tested, with vector acceleration,
  `np.median` and the recorded `first_success_step`.
- Claims corrected in the docs: the 5 mm calibration is empirical, not a
  guarantee (the audit's two counterexamples are test fixtures, as is the
  rigid 10 mm shift that passes); holding each 10 fps action for five ticks
  follows a demonstration to about 0.05 rad, it does not replay it; 21 -> 19
  was a deterministic behaviour change, not run noise; the renderer test
  allows one unit of rasterization difference; dangling links to the deleted
  `docs/plan.md` removed.
- The optional full-assembly tests (`TANGRAM_SLOW=1`) had failed on a stale
  IoU threshold; they assert the benchmark's own criterion and pass.

Final controller and criterion, train seeds 0–9 per figure, 12,000 steps.
Sustained success from the collector (`2026-09-11-oracle-panda.json`) and
success at the horizon from `eval.py` (`2026-09-11-oracle-panda-horizon.json`):

| Figure | At horizon | Sustained | 95% Wilson | Final IoU of successes | Median first success |
| --- | --- | --- | --- | --- | --- |
| square | 8/10 | 8/10 | 0.49–0.94 | 0.934–0.979 | 213 s |
| rectangle | 7/10 | 7/10 | 0.40–0.89 | 0.948–0.979 | 219 s |
| house | 6/10 | 6/10 | 0.31–0.83 | 0.933–0.979 | 221 s |
| cat | 7/10 | 7/10 | 0.40–0.89 | 0.956–0.979 | 219 s |
| all | 28/40 | 28/40 | | | |

The three successes gained over 25/40 come from the settled test: slabs
released 2–4 mm up on a neighbour used to be re-grasped, now they count as
placed when the scorer agrees. Smoothness with the corrected metric: peak TCP
acceleration 2.7–3.3 m/s² on average per episode, 7.2 m/s² at most (one house
episode), peak speed about 0.3 m/s, longest still interval 4.1 s.

Not done, from the audit's recommendations: per-phase timing to speed up
free motion separately from carrying (the negative speed result changed both
at once); a glide duration scaled to the joint distance; measuring slip and
rotation of the slab in the pinch before changing the knob; validation seeds
outside 0–9 for tuning thresholds.

## 2026-09-11: designed scenes, 300 s horizon, controller toward 40/40

Two changes to the task and eleven measured rounds on the reference
controller, all on train seeds 0–9 per figure recorded with
`tools/collect.py --workers 10` (early stop on the first held success).

Task changes:
- Scenes are no longer random draws. `tangram.layout` maps each seed to a
  grid: goal yaw on a 30° grid, goal centre on a 3×3 grid of ±15 mm, packed
  square yaw on a 45° grid and centre on a 3×3 grid of ±30 mm, with strides
  coprime to the grid sizes so 60 seeds visit every value; the dev split adds
  15° of goal yaw, rotations never seen in training. Every episode and result
  row records the scene in degrees and millimetres. Old scenes are not
  comparable; the reference controller measured 28/40 on them.
- Horizon 12,000 -> 15,000 steps (240 -> 300 s). Seven placements take about
  200 s at the only carry speed that keeps the slab in the pinch (rounds 4 and
  5 below), and one recovery costs about 25 s, so 240 s measured the clock.

| Round | Change | Sustained success | Median first success |
| --- | --- | --- | --- |
| 1 | direct carries when clear of the base, glide scaled to joint distance, lift at 0.1 m/s (old scenes) | 28/40 | 205 s |
| 2 | designed scenes; lift back to 0.06 m/s | 30/40 | 213 s |
| 3 | abort phase lowers a badly held slab before letting go; free motion 0.15 m/s | 33/40 | 203 s |
| 4 | carry 0.08 m/s, shorter close/release waits (rejected) | 26/40 | 180 s |
| 5 | carry 0.07 m/s, original waits (rejected) | 27/40 | 190 s |
| 6 | carry 0.06 m/s, horizon 300 s | 38/40 | 205 s |
| 7 | spent retries abandon the piece to the repair pass; up to five repairs | 38/40 | 205 s |
| 8 | stalled waypoints advance within 2 cm in approach/lift too; via point at 0.45 m (rejected: grasps 2 cm off slip) | 35/40 | 200 s |
| 9 | advance only in transit/transfer; via point back at 0.5 m | 38/40 | 205 s |
| 10 | `impratio` 10 (rejected: halves in-pinch creep but stiffens every contact) | 35/40 | 204 s |
| 11 | transit/transfer waypoint counts as reached after 3 s within 2 cm (**final**) | 38/40 | 205 s |

Findings behind the rounds: a slab dropped from carry height lands on its
edge or knob-down and can never be grasped again, so a failed carry now
descends to 1 cm before opening; a carry waypoint held 6 mm short against a
joint stop, with millimetre "progress" every few seconds, defeated the stall
detector for 20 s; in-pinch creep is 4–13 mm over a 13 s carry and tripling
the finger force removes only a third of it; a slab resting on a neighbour's
edge can creep for tens of seconds after the assembly was accepted, so the
finished controller keeps checking the success test and repairs.

Final numbers with the finished controller (a repair no longer starts when it
cannot end before the horizon, and the done state keeps checking the success
test): sustained 38/40 (`2026-09-11-oracle-panda.json`), at the horizon 37/40
(`2026-09-11-oracle-panda-horizon.json`): square 10/10, rectangle 8/10, house
9/10 at the horizon and 10/10 sustained, cat 10/10. House seed 5 was accepted
at 249 s and lost at 276 s because the parallelogram, resting 2 mm up on a
neighbour's edge, crept until its coverage fell under 85%; 24 s were not
enough for a repair.

The two remaining failures (rectangle seeds 7 and 8) are slip cascades: the
yellow small triangle slipped out during a lift and fell on placed pieces,
pushing them 2–4 cm, and the time to fix that did not fit. The next lever is
grasp geometry, which changes the task's contact physics.

## 2026-09-11: workspace-centred scenes and return home (37/40 -> 39/40)

Two observations from watching ten arms collect a rectangle episode each:
the painted silhouette sometimes lay against the robot base, and that is where
the arm failed; and an arm should end its episode at home.

- Workspace. The design's centres put piece centres as close as 0.22 m to the
  base (source) and 0.24 m (goal). The centres are now (0.35, 0.28) m for the
  square goal, (0.35, 0.31) m for the other goals and (0.36, −0.30) m for the
  packed square, chosen so that every piece centre of every train, dev and
  test scene lies between 0.28 and 0.66 m from the base (`tangram.WORKSPACE`,
  720 scenes checked by a test). Closer than 0.28 m the elbow folds against
  its stop and the forearm meets the shoulder column.
- Return home. The finished controller glides to the home configuration and
  holds it, still watching the success test; the collector records four
  seconds after the hold instead of one, so the return is part of every
  demonstration. The controller reads the goal pose off the painted outline,
  so overridden scenes (`--goal-yaw`) get the right certificate.

Same controller otherwise. Sustained 39/40 (`2026-09-11-oracle-panda.json`)
and at the horizon 39/40 (`2026-09-11-oracle-panda-horizon.json`): square
10/10, rectangle 10/10, house 10/10, cat 9/10; median first success 204–209 s;
final IoU of successes 0.909–0.979; peak TCP acceleration 2.7–2.9 m/s² on
average, 4.2 at most; longest still interval in a success 3.7 s. The failure,
cat seed 8 (goal yaw 240°): the sky-blue large triangle slipped out of the
pinch at 52 s, later placements landed 1–2 cm off, and five repairs did not
fit in the time left; the arm held at home for the last 24 s.

### Rounds

Every collection launch writes into its own folder, `data/<figure>/<date-time>/`
(`view.py --record` too), so a session of ten arms is one dated folder of ten
episodes, and `--resume` grows the newest round with more seeds. Export and
report read every round of a figure, newest copy of a seed first. No change
to the episodes themselves.

### Public demonstration sample

80 episodes recorded with ten arms (20 square, 30 rectangle, 30 house; 80/80
solved, median first success 203-204 s; `2026-09-11-hf-demos-panda.json`),
exported with `tools.export --push` to
[murobotics/tangram-square-rectangle-house-panda-80ep](https://huggingface.co/datasets/murobotics/tangram-square-rectangle-house-panda-80ep).
Export now streams frames into LeRobot's encoder threads instead of writing
PNGs first: 6.9 s per episode against 15.2 s, same videos. Datasets are named
after their content unless told otherwise (this one was renamed to the
automatic name; the old id redirects), and the card is a set of tables. The recording
format stays the collector's npz (lossless frames, piece poses, goal,
annotations, everything the scorer and the replay need); the LeRobot dataset
is derived from it, the way Isaac Lab and ManiSkill derive LeRobot datasets
from their HDF5 recordings.

### Tool-calling agent adapter

`agent.py` adds the `agent` system type: a language model drives the arm the
way Robocurve's Inspect Robots agent does, one `move_to` / `move_by` / `done` /
`give_up` tool call per turn, the adapter interpolating the tool point at
0.06 m/s and solving the IK per tick. Review found one bug before any paid
run: the gripper interpolated from the measured opening, and fingers closed on
the 2 cm knob read 0.25, so every carry began by commanding 0.25 and the piece
fell at the first horizontal motion (three pieces, three drops). Interpolating
from the last command fixed the drop. The second finding was the landing
error: 25-30 mm after a carry, all of it gained during the 0.06 m/s descent
(piece-tool offset 2 mm after the carry, 20 mm after the descent). Capping
descents with a closed gripper at 0.03 m/s, the reference controller's
LOWER_SPEED, brings it to 6-7 mm. The prompt's recipe now lifts a piece to
12 cm, carries it 25 cm and lands it flat 7 mm off the point; a test follows
the recipe end to end.
No model has been run yet; the eleven-finding audit style applies here too.

### Scene grid: full coverage (2026-09-12)

The 2026-09-11 design advanced the four grid indices on the same seed
counter, so the joint scene repeated every lcm(12, 9, 8, 9) = 72 seeds (found
by counting distinct scenes in a long round's index: seeds n and n + 72 were
identical to the frame). New mapping: seed n names combination
(1291 n) mod 7776 in mixed radix, every combination once per 7,776 seeds,
tested over the whole train and dev grids. The goal offsets shrink from
±15 mm to ±10 mm because the rectangle's diagonal at yaw 240° with the
(−15, −15) offset reached 0.279 m from the base, 1 mm inside the 0.28 m
elbow limit; at ±10 mm every scene of every figure keeps at least 5 mm of
margin. Seeds map to different scenes than under the previous design, so
per-seed results before this entry are not comparable with later ones.

Measured on the new grid with the collector, 100 consecutive train seeds
per training figure (plus two square and one rectangle top-up seeds so each
holds 100 successes) and 40 cat seeds (`2026-09-12-oracle-panda-grid.json`):
square 100/102, rectangle 100/101, house 100/100, cat 39/40, 339/343 in all;
median first success 198–204 s; final IoU of successes 0.885–0.979, 0.95 or
better in 315 of 339; peak TCP acceleration 2.7–2.8 m/s² on average, 4.8 at
most; longest still interval 11.9 s (square seed 33, holding the last piece
over the goal through two 6 s stall cycles before the re-plan placed it). The
four failures (square 5 and 27, rectangle 87, cat 13) are slip cascades that
ran out of time. The horizon rate from `eval.py` was last measured on the
previous grid (39/40) and is not repeated here.


### Annotation layout: subtask and plan indices (2026-09-12)

The exported datasets carried the plan and subtask annotations in LeRobot
0.6.1's `language_persistent` column, the whole row list of the episode
repeated on every frame, which a collaborator flagged as opaque for a
planner/policy split. The exporter now follows LeRobot's subtask convention:
each level is stored like the task, a text-to-index table under `meta/` and one
integer per frame. `subtask_index` points into `meta/subtasks.parquet` (the
motion in progress) and `plan_index` into `meta/plans.parquet` (the numbered
plan of the episode, constant over it), -1 where absent; the language columns
are gone. `tools/reindex.py` rewrote
`tangram-square-rectangle-house-panda-300ep` in place from its
`episodes.jsonl` (videos untouched; the rewritten parquets are byte-identical
to a fresh export, as the test checks). The 80ep and 330ep sets on the Hub
keep the old columns.

The first conversion kept the oracle's fine sentences, one per motion (reach,
lift, carry, lower, release) with the figure's name in it: 86 distinct
subtasks over 300 episodes, a combinatorial vocabulary rather than 86 skills.
Asked what a human would need, the answer is one decision per piece, which
piece and where, plus recovering from a slip; the motions between are not
verbalised. The oracle now emits the plan line itself as the subtask while a
piece is being placed, `Place the <piece> at the <place> of the outline.`,
with the place named in the outline's own frame so the same words cover any
silhouette, and keeps the recovery sentences and the closing one.
`tools/reindex.py --relabel` mapped the old sentences to the new ones from
the sidecar and the set was re-pushed: 33 distinct subtasks (24 placements
of the 63 possible, 7 pieces by 9 places; 8 recoveries; 1 closing), 8 plans,
a median of 8 subtask segments per episode. The subtask is now exactly the
plan step in progress, so a high-level planner that emits the plan and a
policy conditioned on the current line share one vocabulary.
