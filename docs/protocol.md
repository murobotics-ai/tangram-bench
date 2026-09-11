# tangram-packed-v6

Public pilot with four silhouettes, two simulated embodiments and two observation
contracts (state or pixels). v6 replaces v5: three training figures and one
held-out figure, pixel observations, a 12,000-step horizon and a demonstration
format. Do not pool results across protocols. It does not claim robust
manipulation, secret-test generalization or sim-to-real.

## Tasks and splits

Seven pieces start as a packed square. Source center is uniform in
x ∈ [0.28, 0.36], y ∈ [−0.30, −0.24] metres, with uniform yaw in [−π, π].
The original piece scale, 0.5 mm edge inset and 2 × 2 × 4 cm grasp knobs are fixed.
Slabs are 5 mm thick; the nominal square side is 0.20√2 m. All pieces stay face up.

`shapes.py` defines `silhouettes-v2`: square, rectangle, house and cat. The house
is the classic tangram house (body, overhanging roof, chimney), mirrored so the
parallelogram needs no flip. Coordinates are in 10 cm units with `sqrt(2)`
diagonals; each certificate lists per piece an eighth-turn rotation and the
position of its first vertex, verified for coverage, area, nonoverlap and no
reflection. The nominal contours may be concave. Inset pieces have a geometric
IoU ceiling of about 0.9788. Certificates are evaluator/reference assets, not
policy observations. `uv run -m shapes` checks the corpus. v2 replaces the v1
pentagon house; house results from v1 are not comparable.

The goal has an independent seeded translation/yaw. Its center is
(0.32, 0.20) m for square and (0.32, 0.27) m for other figures, with ±0.015 m jitter
on each axis. Larger figures are shifted away from the source. Geometric validity
and successful settling at a certificate do not prove that a robot can execute it.

Suite membership: train = square/rectangle/house; dev = the same three figures on
unseen seeds; test = cat, the held-out figure. Seed ranges are [0,100000),
[100000,200000), [200000,300000), respectively. Demonstrations come from train
seeds only, so dev measures execution on seen figures at unseen poses and test
measures generalization to an unseen silhouette. Episodes cycle through the
suite's figures; use a multiple of the suite size for equal counts. `--offset`
selects the seed range. `--target NAME` runs one diagnostic figure and is
recorded explicitly. The held-out figure is publicly inspectable. One held-out
animal is not broad shape-family generalization.

## Physics and horizon

Panda and PiPER use the pinned Menagerie models, with existing gripper gains,
actuator limits and contact settings in `env.py`. Both CPU and Warp begin with
100 CPU settling steps. Physics is 500 Hz; commands execute at 50 Hz.

Default horizon: 12,000 actions / 240 simulated seconds; the reference controller
needs about 200 seconds. The full horizon runs even after transient success. A fresh policy is constructed per episode with its seed.
Source/goal sampling and seeds are fixed before execution. Policies must seed
local RNGs; shared globals can make batching affect behavior.

The physical reference controller `examples/oracle.py` (Panda only) assembles
any figure from its certificate through contact. Its success rate per figure is
measured with `tools/collect.py` and reported in the README; it is a data
generator and an execution ceiling, not a leaderboard entry. The PiPER wrist
limits can restrict overhead grasps; PiPER has no validated controller.

## Prompt, annotations and observations

The task prompt names the figure: `Solve the tangram puzzle to assemble the
house.` (`shapes.prompt(target)`). It is the only language a policy receives
at evaluation; it appears in the viewer's top panel, in `observation["prompt"]`
and in every result and trajectory. `--prompt TEXT` replaces it for all
figures and changes comparison identity (results record `prompts` per figure).
Policies may ignore it. Remote adapters can declare `use_prompt: false`.

Demonstrators may expose language annotations: `policy.plan`, a numbered list
of steps for the episode (one sentence per piece, e.g. `Place the orange large
triangle at the right of the house.`); `policy.step`, the 1-based step in
progress; and `policy.subtask`, a finer sentence for the current motion
(`Carry the orange large triangle to the right of the house and align it.`).
Evaluation trajectories and recorded demonstrations store `plan` per episode
and `step` and `subtask` per step or frame; the viewer shows the current one
under the prompt as `SUBTASK 3/7: ...` while replaying. None of it is given to
a policy at evaluation.

`tools/export.py` writes them the way `lerobot-annotate` does (LeRobot 0.6.1,
`lerobot.datasets.language`): a `language_persistent` column of rows
`{role, content, style, timestamp, camera, tool_calls}` with `subtask` rows at
every change and `plan` rows at every step boundary holding the numbered list
of steps still to do, an empty `language_events` column, and both declared in
`meta/info.json`. The LeRobot task string stays the figure prompt so training
and evaluation see the same language; `--task subtask` appends the annotation
for subtask-conditioned training.

Observations contain independent copies:

- `robot`, `target`, `prompt`: embodiment, figure identifier and task text.
- `time`: elapsed simulated seconds, excluding reset settling.
- `qpos`, `qvel`: arm plus two finger joints (9 for Panda, 8 for PiPER).
  Revolute positions/velocities are radians/radians per second; finger slides use metres.
- `tcp_pos` (3,), `tcp_mat` (3,3): tool position and tool-to-world rotation.
- `pieces` (7,7): body-origin xyz and unit wxyz quaternion.
- `piece_velocities` (7,6): world linear xyz and local angular velocity.
- `goal` (V,2): ordered contour vertices in world xy; V depends on the figure.

Piece order is two large triangles, medium triangle, two small triangles, square,
parallelogram. World/base frame uses table z = 0. Local piece geometry is public.

`--obs state` (default) gives the fields above. `--obs pixels` adds
`images`: `{"top": (240,320,3), "wrist": (240,320,3)}` uint8 RGB, rendered with
`mujoco.Renderer` only on ticks where the policy is queried. The goal silhouette
is painted on the table in every camera, so pixels carry the target; state
fields remain available and a pixel policy is expected to declare
`access = "pixels"` if it ignores them. The context camera is an inspection
view only. The observation contract is part of the result identity.

## Actions and adapters

One action is absolute arm joint targets in radians plus gripper opening
(0 closed, 1 open): width 8 for Panda, 7 for PiPER. A policy can return one action
or a `(K, action_dim)` chunk, 1 ≤ K ≤ `--max-chunk` (default 1). The whole chunk is
validated before its first command. Commands execute open loop at 20 ms intervals;
inference resumes when the queue empties. Unconsumed predictions at the horizon
are discarded. Arm targets pass through a 0.04 rad/action slew limit.

Invalid shapes, nonfinite values, joint-limit violations and gripper values outside
[0,1] are policy errors. No direct object actions, attachments or pose assignments
are allowed. Policy modules are trusted local code, not a security sandbox.

`--policy FILE` uses `Policy(robot, seed).act(observation)`. `--system JSON` selects
an adapter instead. Local configurations specify a module, optional checkpoint
and constructor kwargs. Checkpoint/module paths resolve relative to that file and
are hashed. Loading weights and converting observations/actions belong to the
policy implementation; a checkpoint alone does not specify an architecture.

HTTP sends robot, seed, observation, base64 PNG images when pixels are on, the
joint-action contract and adapter kwargs; the server returns
`{"actions": [[...], ...]}`. The OpenAI adapter uses the
[Responses API](https://developers.openai.com/api/docs/api-reference/responses/create)
with `text.format` JSON-schema structured output; the Anthropic adapter uses
[Messages](https://platform.claude.com/docs/en/api/messages/create) with
`output_config.format`. Both send the instructions as the system prompt, then the
images labelled `Image 1: top camera` and `Image 2: wrist camera` before the
observation JSON (state rounded to 4 decimals), and both accept an optional
`effort` (reasoning effort) and `use_state` / `use_images` switches; a state-free
run declares `access = "pixels"`. Model IDs are supplied explicitly or resolved
from `model_env`; API keys are read from environment variables, loaded from a
git-ignored `.env` at the repository root if present (shell values win), and never
included in transcripts. Rate limits and server errors are retried up to
`retries` times (default 3) honouring `Retry-After`; refusals, truncated output
and malformed JSON are policy errors. Authenticated provider availability depends
on the user's account; automated adapter tests use mocked responses.

`--max-inference-calls` defaults to the horizon (12,000) per episode. Exhaustion is a failed
attempt. `--max-output-tokens` caps provider output per request (default 4,096);
an adapter cannot exceed it. HTTP/provider calls have a configured timeout of at
most 300 seconds. In-process policies have no watchdog. Simulator time waits for
inference; this is not a real-time latency benchmark. Calls, simulated time and
wall time are distinct. Provider requests/responses and returned token usage are
saved as audit sidecars. Remote pricing and missing usage fields are not inferred.

## Scoring

Project the actual inset slab polygons using full body quaternions. For union U
and goal G, IoU = area(U ∩ G) / area(U ∪ G). Overlap is
(sum of footprint areas − area(U)) / area(G). A timestep passes only if:

- IoU ≥ 0.95 and overlap < 0.01;
- every body origin is within 2 mm of half-thickness above the table;
- every local z axis is within 5° of upward vertical;
- every piece moves below 0.01 m/s linearly and 0.1 rad/s angularly.

Success requires the final 25 consecutive post-action timesteps to pass. Reset
never contributes. Gripper-empty completion is not required. First sustained
success is diagnostic even if the policy subsequently disturbs the assembly.

Report success rate over all attempts with a 95% Wilson interval. Counts per
figure are balanced in suite runs whose episode count is a multiple of the suite size. The small corpus does not support a population
claim over arbitrary unseen figures; the interval describes scene sampling,
not training variance or family-level uncertainty. Policy errors count as failures.
Final IoU averages cover only full-horizon episodes, with their denominator explicit.

Diagnostics include coverage, number of projected pieces with ≥95% of their area
inside the goal, initial/best/final IoU, hold length, physical gates, failure reasons,
executed steps and inference latency. Projection coverage alone is not success.
Malformed scoring evidence raises an error. Privileged oracle controllers declare
`access = "oracle"`; their results are grouped separately from state policies.

## Demonstrations

`uv run -m tools.collect --target NAME --episodes N` records seeded episodes of
one figure with any policy (default: the reference controller) as
`data/NAME/episode-SEED.npz`: top/wrist images, `qpos` as state, the commanded
action, piece poses, goal and identity, at `--fps` frames per second (default 10;
must divide 50). A frame's action is the last command of its 50/fps-tick
interval, so a policy that emits one action per frame and holds it for the
interval reproduces the demonstration. Episodes stop one second after a held
success unless `--full-horizon`. Failed attempts are listed in `index.jsonl`
and skipped unless `--keep-failures`. `view.py --teleop --record DIR` saves
teleoperated episodes in the same format. `uv run -m tools.export FOLDERS --out
DIR` converts folders to one LeRobot v3 dataset (`observation.images.top`,
`observation.images.wrist`, `observation.state`, `action`, task = prompt) with an
`episodes.jsonl` sidecar. `uv run -m tools.train --policy smolvla` wraps
`lerobot-train` with the repository paths: pretrained weights download to
`checkpoints/` (the Hub cache, `HF_HUB_CACHE`) and the run is written to
`outputs/train/NAME/`. `examples/lerobot_policy.py` runs a LeRobot checkpoint
as a pixel policy, repeating each predicted action 50/fps ticks. Evaluation
results default to `outputs/runs/`; `data/`, `checkpoints/` and `outputs/` are git-ignored, while
`results/` holds the versioned summaries.

## Artifacts, verification and history

`result.json` has a versioned schema, UTC dates, exact seeds/figures/prompt/budgets,
robot/backend, model hash, package versions, policy/checkpoint/configuration hashes,
source hashes, metadata, status, summary and episode records. `--policy-artifact`
adds files to the hash manifest; `--policy-metadata` adds provenance. Declared
sources/artifacts are checked again before completion. Undeclared dependencies
and mutable remote model aliases remain reproducibility limitations.

`result.artifacts/` contains the initial manifest, compiled `model.mjb`, fsynced
episode journal, optional provider audit JSON and one NPZ per episode. NPZ records
reset plus every valid post-action state, requested actions, applied actuator
controls, time and prompt/target identity. States have one more row than actions.
Existing result/artifact paths are never overwritten.

Policy errors stop that policy and score false; other worlds continue. Failed
worlds hold their last valid command but their later states are not recorded as
that policy's execution. Setup/physics/source-change errors or interruption make
the run invalid, with null aggregate and a nonzero exit code. Valid prefixes and
finished episodes are retained when possible. Process kill may lose the current
batch; there is no automatic retry/resume or favorable averaging of partial runs.

`uv run -m tools.results verify result.json` checks hashes, identities, time grid,
seed/figure manifest and horizon, then recomputes scores/summary without a policy.
Use the recorded scoring source revision. This checks integrity and scoring,
not tamper-proof execution. Move the JSON and artifact directory together.

`compare A.json B.json` verifies both and requires matching protocol, dataset,
robot/backend, Python/packages, source/model hashes, prompt, targets, exact seeds,
horizon, batching, chunk/call/token limits and access level. Results are paired by
seed. A 10,000-resample scene bootstrap (seed 0) reports B − A with wins/losses/ties.
A degenerate interval on constant outcomes does not establish equivalence.

`view.py --result result.json --episode 0` reconstructs the recorded state using
its saved model; `view.py --demo data/house/episode-3.npz` replays a recorded
demonstration at its frame rate. Space pauses, arrows seek one second, Home/End
jump to endpoints. Neither reruns physics or the policy. The prompt and the
current annotation stay above the scene.

`uv run -m tools.history outputs/runs --out outputs/runs/history.html` generates a self-contained
local history viewer. It separates compatible cohorts, shows date/checkpoint/score,
and offers top-down episode playback. Display sampling defaults to every 10th
step; scoring still uses every state. This is a reconstruction, not recorded camera
video. Historical results with other scoring source versions are labeled unverified
and excluded from the verified plot. It is not a hosted or private-test leaderboard.
