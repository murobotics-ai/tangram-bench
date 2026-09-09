# Plan: from pilot to a generalization benchmark

Goal. One question, measured exactly: **does a robot policy generalize to a
silhouette it never trained on?** Train on many silhouettes, test on held-out
structural families, score with exact geometry.

Style. Few files, one command per job, no framework, no config sprawl. Everything
runs on a laptop with `uv run`. A reader should understand the whole benchmark
in an afternoon and add a policy in an hour.

What exists today: exact-state observations, joint-action interface, four
silhouettes, one oracle demonstration on one seed, result verification and
comparison tools. What is missing is below, in dependency order.

## Milestones

| | Milestone | Proof |
| --- | --- | --- |
| M1 | A fine-tuned VLA scores nonzero on the execution track | One results file with Wilson interval |
| M2 | Reasoning track with a generated corpus and held-out families | Planner baseline beats VLA on unseen families |
| M3 | Community can submit and compare | Leaderboard file, three external entries |

M1 answers "does the benchmark produce signal". Without it nothing else matters.

## 1. Corpus generator (`corpus.py`)

Replace the four hand-authored silhouettes with a generated corpus.

- Enumerate silhouettes reachable by the seven pieces on the 10 cm lattice,
  quarter-turn rotations only, no flips. Grow unions piece by piece on the
  lattice; keep unions that are simple polygons without holes.
- Canonicalize under rotation and translation, deduplicate, record every
  certificate per silhouette. A backtracking exact-cover search over the lattice
  finds all certificates in well under a second per silhouette (prototype done:
  house 32, rectangle 72, cat 8).
- Assign each silhouette to a structural family from outline features: vertex
  count, concavity count, symmetry group, bounding-box aspect. Splits are by
  family with a fixed hash, so "unseen" means an unseen family, never a
  neighbouring variant.
- Ship `silhouettes-v2.json`: outline, family, split, certificate count, and
  certificates for train and dev only. The metric is IoU against the outline
  and needs no certificate, so test certificates stay private without hurting
  anyone's ability to verify a result.
- Command: `uv run corpus.py --train 500 --dev 50 --test 50 --seed 0`.
- Keep `validate()` as the single gate every silhouette passes.

Depends on nothing. Everything downstream depends on it.

## 2. Pixel observations (`env.py`)

Cameras `top`, `wrist` and `context` are already defined but only rendered in
the viewer. Make them observations.

- Add `images` to the observation dict: `{"top": (H,W,3), "wrist": (H,W,3)}`
  uint8, fixed resolution, rendered with `mujoco.Renderer` only when the policy
  is queried, not every physics step.
- The goal silhouette is already drawn on the table, so pixels carry the
  target. State fields remain for the state track.
- Warp backend: sync `qpos` to CPU `MjData` before rendering. Already done for
  the state path.
- Protocol name becomes `tangram-pixel-v1`; `--obs state|pixels` selects the
  input and is recorded in the result identity.

Depends on nothing. Required by every VLA.

## 3. Two tracks (`eval.py`, `benchmark.py`)

- `--track execution`: observation includes `targets` (7,7), the goal pose per
  piece, and the rendered shadow is coloured per piece. Measures precision and
  long horizon only.
- `--track reasoning`: silhouette only. Measures assignment plus execution.
- The difference between the two numbers for the same policy is the headline:
  what is lost by having to infer the plan.
- Track is part of the result identity; `compare` refuses to mix tracks.

Depends on 1 for reasoning-track targets across many silhouettes.

## 4. Partial metrics (`tangram.py`, `benchmark.py`)

Binary success at IoU 0.95 stays. Add signal below it so early policies can be
ranked:

- `pieces_placed`: count of pieces with `piece_coverage ≥ 0.9` and no overlap.
- `assignment_score`: best match between observed piece poses and any
  certificate of the silhouette, as a fraction of pieces within 1 cm and 10°.
  Uses all certificates from 1, so any valid decomposition scores fully.
- `final_iou`, `first_success_seconds`, `final_hold_steps` already exist.
- `summarize` reports all of them with Wilson intervals; `success` stays the
  leaderboard number.

Depends on 1 for certificates.

## 5. Oracle across seeds and robots (`examples/oracle.py`)

The only validated demonstration is one seed, one silhouette, Panda, four
times the default horizon. The oracle must become a reliable data generator.

- Merge the house controller's grasp selection, IK checks and transit into
  `oracle.py` so one controller serves every silhouette from its certificate.
- Run 100 seeds × all train silhouettes × both robots. Publish the success
  table in the README. Fix the top failure modes until Panda exceeds 90%.
  PiPER wrist limits are a known risk; report honestly if it stays lower.
- Measure time-to-success and set `DEFAULT_STEPS` from the 95th percentile,
  likely 6000 to 12000 rather than 3000.
- Delete `examples/scripted.py`; it is superseded.

Depends on 1. The hardest engineering item in the plan.

## 6. Demonstration dataset (`tools/collect.py`)

- Run the oracle over the train split with pixels on; save each episode as
  images, state, actions, prompt and silhouette id.
- Write LeRobot format directly, since openpi, LingBot-VLA and XPolicyLab all
  consume it. Keep a native npz too for people without those stacks.
- Target 100 episodes per train silhouette. Simulation is cheap: publish the
  command, not only the download, so anyone can regenerate or scale it.
- Command: `uv run -m tools.collect --split train --episodes 100 --out data/`.

Depends on 2 and 5.

## 7. Policy adapters (`adapters.py`)

- Add `xpolicylab`: websocket client that maps `images.top → cam_head`,
  `images.wrist → cam_left_wrist`, `qpos → left_arm_joint_state`, prompt →
  instruction, and joint-position actions back. One adapter gives access to
  the 40+ policies already integrated there, LingBot-VLA and π0.5 included.
- Add `openpi`: same idea for the openpi websocket server.
- Keep `local` and `http`. Keep `openai` and `anthropic` only if a planner
  baseline uses them; otherwise remove.

Depends on 2.

## 8. Baselines (`examples/`)

Three, reported from day one, each labelled by access level:

- `oracle`: certificate given. Ceiling for execution.
- `planner`: silhouette only; the exact-cover search from 1 picks a
  certificate, the oracle controller executes it. Access `state-planner`.
  This is the reasoning-track ceiling for a modular system.
- `vla`: one fine-tuned open model, π0.5 LoRA or LingBot-VLA, trained on 6,
  evaluated on both tracks and on train, dev and test families.

Expected picture: oracle high, planner close behind, VLA nonzero on execution
and near zero on unseen reasoning families. That gap is the paper.

Depends on 3, 4, 6, 7.

## 9. Protocol and leaderboard (`docs/protocol.md`, `LEADERBOARD.md`)

- Bump protocol: pixels, tracks, corpus version, horizon, metrics.
- Fixed test seeds and families. Results files carry policy digest, track,
  corpus version and protocol; `verify` recomputes every score from the saved
  trajectory, which already exists.
- `LEADERBOARD.md` updated by pull request with the results file attached.
  No server. A maintainer runs `verify` before merging.
- Every entry reports train, dev and test family success side by side, so
  memorization shows up as a train–test gap.

Depends on 3, 4.

## 10. Tests and CI (`tests/`)

- Corpus: every silhouette validates; splits are disjoint by family; test
  certificates absent from the shipped file.
- Pixels: shape, dtype, determinism across backends for the same seed.
- Tracks and metrics: execution-track observation contains targets; reasoning
  does not; `assignment_score` is 1 at any certificate and 0 far away.
- Oracle: one short episode per robot on a fixed seed, kept under a minute.
  The 12000-step house test moves to a nightly job.

Depends on each item as it lands.

## 11. Docs (`README.md`)

Rewrite around five commands:

```
uv run tools/prepare.py            # robot assets
uv run corpus.py                   # silhouettes and splits
uv run -m tools.collect            # demonstrations
uv run eval.py --system my.json    # evaluate a policy
uv run -m tools.results compare    # compare two results
```

Move design history to `tangram-bench.md`; keep the README operational.

## 12. Real world (after M3)

Out of scope for v1. The design already makes it cheap: pieces are printable
with the knob, the knob top is a fiducial, reset is "put it back as the
square". A pose tracker feeding the same scorer is the whole integration.

## Dependency graph

```
1 corpus ──┬──> 3 tracks ──┐
           ├──> 4 metrics ─┤
           └──> 5 oracle ──> 6 demos ──> 8 baselines ──> 9 leaderboard
2 pixels ──┬──> 6 demos                      ▲
           └──> 7 adapters ──────────────────┘
10 tests and 11 docs land with each item
```

Critical path for M1: 2 pixels, 5 oracle on the four current silhouettes,
6 demos, 7 adapters, one VLA fine-tune on the execution track. The corpus can
land in parallel and is required for M2.

## Risks, stated plainly

- Oracle reliability across seeds is unproven. If it stays under 90% the
  dataset is noisy and every downstream number inherits that noise.
- The reasoning track may read zero for every VLA. That is an expected and
  publishable result only if the execution track and partial metrics give a
  nonzero ranking next to it.
- Corpus size is unknown until enumerated. If the lattice yields few families,
  add a second lattice scale or allow half-unit offsets before widening the
  piece set.
- Rendering cost with Warp batching may dominate evaluation time. Render only
  on inference calls and measure before optimizing.
