# Experiment log

One entry per experiment, newest last, negative results included. Summaries of
the runs live next to this file as dated JSON; raw trajectories stay in `outputs/runs/`
(not versioned). Every number in the README points here.

## 2026-09-10: reference controller across seeds (baseline 2/20 -> 14/40)

Setting: Panda, train seeds 0-9 per figure, 12,000 steps, default physics.
File: `2026-09-10-oracle-panda.json`.

| Figure | Success | 95% Wilson |
| --- | --- | --- |
| square | 3/10 | 0.11–0.60 |
| rectangle | 5/10 | 0.24–0.76 |
| house (v1 pentagon) | 2/10 | 0.06–0.51 |
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

## 2026-09-10: classic house replaces the pentagon house (silhouettes-v2)

The v1 "house" was a pentagon (4x1 body plus a triangular roof). It is now the
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
