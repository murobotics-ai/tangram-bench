# Mu-Bench

**A benchmark for world-aware robot learning through physical construction.**

## Essence

Mu-Bench asks a simple question:

> Can a robot understand how the physical world will change before it acts?

We want to evaluate agents that do more than map observations and instructions directly to actions. A capable agent should be able to imagine a future state, choose a physically valid action, observe what actually happened, and correct itself when reality differs from its prediction.

Mu-Bench begins with Tangram: seven pieces, one work surface, and a large space of possible goals. The world is deliberately small, but the challenge still requires perception, geometric reasoning, planning, precise manipulation, memory, and recovery.

## Vision

Our vision is to build a progressive benchmark for physical intelligence.

Each Mu-Bench domain should be simple enough that states, actions, intermediate predictions, and failure modes can be measured precisely, while remaining rich enough to expose capabilities that current robot-learning systems lack.

Tangram is the first domain, not the limit. Future domains may introduce 3D construction, occlusion, regrasping, packing, kitting, assembly, and increasingly complex contact. The common principle is that the agent must construct a verifiable physical future through action.

Mu-Bench is not intended to answer only, “Which policy has the highest success rate?” It should help answer:

- What did the agent understand about the goal?
- What physical outcome did it predict?
- Was its plan geometrically and physically feasible?
- Did the failure come from reasoning or execution?
- Can the agent detect an error and recover without restarting?
- How does prediction error compound over a long horizon?

## Why

Most manipulation benchmarks reduce an episode to final task success. That number is useful, but it conflates several different capabilities: perception, goal interpretation, planning, world modeling, control, and recovery.

When a robot fails, we often cannot tell whether it formed the wrong plan or executed the right plan poorly. When it succeeds, we cannot tell whether it understood the task, reproduced a familiar trajectory, or reached the goal through repeated trial and error.

Tangram gives us an unusually clean diagnostic environment:

- The state is compact and measurable: the position and orientation of seven rigid pieces.
- A top-down camera makes the relevant world state nearly observable.
- Predictions can be scored in millimeters, degrees, overlap, and geometric feasibility rather than only pixel similarity.
- Every placement changes what remains possible, creating genuine long-horizon dependencies.
- The parallelogram introduces chirality: some solutions require a physical 3D flip, not a 2D rotation.
- Multiple valid decompositions can test whether evaluation rewards the physical goal rather than one memorized solution.
- Simulation can scale evaluation, while real hardware can expose calibration, friction, contact, and grasping errors.

Tangram is narrow by design. Mu-Bench does not claim that solving it is equivalent to general manipulation. Its value is diagnostic: it makes a specific set of physical reasoning capabilities observable and comparable.

## How

### 1. Separate reasoning from execution

Mu-Bench will evaluate three complementary modes:

1. **Planning:** the agent proposes piece-to-pose assignments and a sequence of subgoals; a reference controller executes them.
2. **Execution:** the agent receives a valid plan and must perceive and manipulate the pieces accurately.
3. **End-to-end:** the agent interprets the goal, plans, acts, observes the result, and recovers when necessary.

The difference between these modes reveals whether a system is limited by reasoning, control, or their interaction.

### 2. Record reasoning-rich demonstrations

We will collect live-annotated demonstrations of operators solving Tangram puzzles. Each episode will align:

- camera observations;
- robot and object states;
- actions and subgoals;
- the operator's spoken reasoning;
- predicted intermediate states;
- corrections, backtracking, and failure recovery.

The goal is not merely to record successful trajectories. We want data that captures why a move was selected, what the operator expected to happen, and how the plan changed after an error.

### 3. Train policies with memory and a world model

We will develop a policy architecture inspired by π0.7, augmented with memory and a world model. The model should maintain task context across the full construction, predict the consequences of candidate actions, and use discrepancies between predicted and observed states to update its plan.

Classical systems—segmentation, combinatorial solving, and visual servoing—will be included as explicit baselines and partial oracles. A learned system should not receive credit merely for solving a puzzle that a deterministic pipeline can already solve reliably; the comparison should reveal what learning adds and where it still falls short.

### 4. Evaluate capabilities, not only completion

The first benchmark will include four diagnostic suites:

- **Base:** nominal construction from controlled initial states.
- **Chiral:** goals that require correctly representing and executing a physical flip.
- **Recover:** partially completed scenes containing mistakes that must be detected and repaired.
- **Imagine:** prediction of future states and counterfactual action outcomes before execution.

Metrics will include:

- final silhouette intersection-over-union and progress after each action;
- position and orientation error per piece;
- geometric validity of the proposed plan;
- disturbance of previously placed pieces;
- number of actions, corrections, and reversals;
- prediction error at increasing horizons;
- recovery success and recovery cost;
- performance across seen and unseen goals, configurations, instructions, and visual conditions.

### 5. Connect simulation and reality

Simulation will provide scale, controlled interventions, and statistically meaningful evaluation. A standardized real setup will test whether simulated results predict deployment performance.

Mu-Bench will specify piece geometry, observation and action spaces, camera configuration, control frequency, initial-state distributions, reset procedures, success criteria, and robot embodiment. Results from suction and parallel-jaw grippers will be reported separately rather than hidden in one aggregate score.

## What

The first release of Mu-Bench will deliver:

- a standardized Tangram task specification for simulation and real hardware;
- a corpus of procedurally generated puzzles and explicit generalization splits;
- reasoning-rich human and robot demonstrations;
- planning, execution, end-to-end, recovery, and prediction protocols;
- classical and learned baselines;
- metrics and evaluation tooling;
- a reproducible sim-to-real validation protocol;
- an open leaderboard with per-capability results and failure analysis.

The target task is ambitious but concrete:

> Given a valid Tangram goal, solve it from a novel initial configuration, predict the consequences of each action, and recover from mistakes without restarting.

## What is actually hard right now?

We are building the first Mu-Bench domain around Tangram because it requires physical reasoning while remaining precisely measurable. Our immediate objective is to train a model that can solve any valid Tangram puzzle, not only reproduce figures seen during training.

The hard part is connecting reasoning to action over the full episode. We will record live-annotated demonstrations in which an operator explains the plan, expected physical outcome, uncertainty, and corrections while solving each puzzle. Using this data, we will develop a π0.7-inspired policy with memory and a world model that can retain context, imagine action outcomes, execute a plan, and revise it when the world changes unexpectedly.

Mu-Bench starts small on purpose. Seven pieces are enough to ask whether a machine can imagine a physical transformation—and make it real.
