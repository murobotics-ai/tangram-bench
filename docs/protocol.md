# tangram-packed-state-v5

Public pilot with four silhouettes and two simulated embodiments. This protocol
replaces the square-only v4 observation/evaluation contract; do not pool results.
It does not claim robust manipulation, secret-test generalization or sim-to-real.

## Tasks and splits

Seven pieces start as a packed square. Source center is uniform in
x ∈ [0.28, 0.36], y ∈ [−0.30, −0.24] metres, with uniform yaw in [−π, π].
The original piece scale, 0.5 mm edge inset and 2 × 2 × 4 cm grasp knobs are fixed.
Slabs are 5 mm thick; the nominal square side is 0.20√2 m. All pieces stay face up.

`shapes.py` defines `silhouettes-v1`: square, rectangle, house and cat. Each has
seven SE(2) transforms of the original pieces, verified for coverage, area,
nonoverlap and no reflection. The nominal contours may be concave. Inset pieces
have a geometric IoU ceiling of about 0.9788. Certificates are evaluator/reference
assets, not policy observations. `uv run -m shapes` checks the corpus.

The goal has an independent seeded translation/yaw. Its center is
(0.32, 0.20) m for square and (0.32, 0.27) m for other figures, with ±0.015 m jitter
on each axis. Larger figures are shifted away from the source. Geometric validity
and successful settling at a certificate do not prove that a robot can execute it.

Default suite membership: train = square/rectangle; dev = house; test = cat.
Seed ranges are [0,100000), [100000,200000), [200000,300000), respectively. Episodes
cycle through the suite with equal counts per figure; `--offset` selects the seed
range. `--target NAME` runs one diagnostic figure and is recorded explicitly.
Figures are disjoint across these small public splits, but test figures are
publicly inspectable. One held-out animal is not broad shape-family generalization.

## Physics and horizon

Panda and PiPER use the pinned Menagerie models, with existing gripper gains,
actuator limits and contact settings in `env.py`. Both CPU and Warp begin with
100 CPU settling steps. Physics is 500 Hz; commands execute at 50 Hz.

Default horizon: 3,000 actions / 60 simulated seconds. The full horizon runs even
after transient success. A fresh policy is constructed per episode with its seed.
Source/goal sampling and seeds are fixed before execution. Policies must seed
local RNGs; shared globals can make batching affect behavior.

The physical reference controllers remain experimental. Tests verify rest/reset,
static solution states, and one physical house assembly with Panda using
`examples/house.py`, development seed 100076 and a 12,000-step horizon. This
selected-scene oracle demonstration does not establish a success rate across
seeds, standard-horizon completion or contact-rich CPU/Warp parity.
The PiPER wrist limits can restrict overhead grasps. `examples/oracle.py` records
failures rather than treating certificate placement as executed manipulation.

## Prompt and observations

Default prompt: `Assemble the tangram to match the silhouette using all seven pieces.`
The same text appears in the rectangular top panel, policy observation and saved
result/trajectory. `--prompt` changes it and therefore changes comparison identity.
Policies may ignore it. Remote adapters can declare `use_prompt: false`; that
choice is recorded in their system configuration.

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
Current policy input is privileged state. Top/wrist/context camera panels are
inspection views, not pixel observations. The silhouette is drawn consistently
in those views using exact triangle geometry, without collision geometry.

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

HTTP sends robot, seed, observation, joint-action contract and adapter kwargs;
the server returns `{"actions": [[...], ...]}`. OpenAI uses the
[Responses API](https://developers.openai.com/api/docs/guides/structured-outputs);
Anthropic uses [Messages](https://platform.claude.com/docs/en/api/http/messages/create).
Each provider adapter constructs its own request and parses its own response
before converting JSON actions to arrays. Exact model IDs are supplied explicitly
or resolved from `model_env`. API keys are read from environment variables and
never included in transcripts. Provider refusals/malformed output are failures.
Authenticated provider availability depends on the user's account; automated
adapter tests use mocked responses, not claims of live model performance.

`--max-inference-calls` defaults to 3,000 per episode. Exhaustion is a failed
attempt. `--max-output-tokens` caps provider output per request (default 2,048);
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
figure are balanced in suite runs. The small corpus does not support a population
claim over arbitrary unseen figures; the interval describes scene sampling,
not training variance or family-level uncertainty. Policy errors count as failures.
Final IoU averages cover only full-horizon episodes, with their denominator explicit.

Diagnostics include coverage, number of projected pieces with ≥95% of their area
inside the goal, initial/best/final IoU, hold length, physical gates, failure reasons,
executed steps and inference latency. Projection coverage alone is not success.
Malformed scoring evidence raises an error. Privileged oracle controllers declare
`access = "oracle"`; their results are grouped separately from state policies.

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
its saved model. Space pauses, arrows seek one second, Home/End jump to endpoints.
It does not rerun physics or the policy. The prompt stays above the scene.

`uv run -m tools.history runs --out runs/history.html` generates a self-contained
local history viewer. It separates compatible cohorts, shows date/checkpoint/score,
and offers top-down episode playback. Display sampling defaults to every 10th
step; scoring still uses every state. This is a reconstruction, not recorded camera
video. Historical results with other scoring source versions are labeled unverified
and excluded from the verified plot. It is not a hosted or private-test leaderboard.
