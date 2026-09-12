"""Tool-calling language-model agent: Cartesian targets, one tool call per turn.

The `agent` system type gives a language model a tool-calling interface to the
arm: every turn it receives the proprioceptive state, the cameras and the last
tool result, and answers with exactly one tool call. `move_to` and `move_by`
name a destination for the tool point (metres, degrees, gripper opening); the
adapter interpolates from the observed pose at a fixed safe speed, solves the
damped IK of `teleop.py` for every control tick and hands `eval.py` the joint
chunk the protocol expects. `done` and `give_up` end the model's part; the arm
then holds still for the rest of the horizon. Unreachable targets are rejected
with a reason and three rejections in a row are a policy error. Every request
and reply is logged next to the trajectory with images replaced by digests.
"""

import base64
import hashlib
import json
import math
import os
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mujoco
import numpy as np

from env import DT, HOME, SUBSTEPS, make_model
from tangram import NAMES, THICKNESS
from teleop import Teleop

OPENAI_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
TICK = DT * SUBSTEPS
# Tool-point workspace box (m): the table is 2 m square around x = 0.3; z is the
# fingertip height above the table.
WORKSPACE = np.array([[0.15, 0.80], [-0.60, 0.60], [0.01, 0.60]])
GRASP_HEIGHT = 0.025  # Tool point height that pinches the knob (examples/oracle.py).
RELEASE_HEIGHT = 0.03  # Tool point height at which a held piece rests on the table.
LOWER_SPEED = 0.03  # m/s cap for descents with the gripper closed: faster ones creep the slab
# out of the pinch by 15-20 mm before it touches down (examples/oracle.py LOWER_SPEED).
MAX_REJECTIONS = 3
MAX_WAYPOINT_RESIDUAL = 0.004  # m; the IK must reach the target this closely.
MAX_ANGLE_RESIDUAL = np.deg2rad(2)

SYSTEM = """You are controlling a Franka Panda arm in a MuJoCo simulation through tool \
calls. The task: assemble the seven tangram pieces into the silhouette painted on the \
table. Each observation message gives you the tool point pose, the gripper opening and \
camera images. Work toward the goal in small, deliberate motions and re-check the \
observation after every motion. Every move tool call must include a `note`: in one or \
two sentences, say what you observe and why you chose this motion; a human reads these \
notes to follow you. Respond with exactly one tool call per turn. When the goal is \
achieved call done; if it cannot be achieved call give_up. Both ask what you wish you \
had known from the start, so note what you learn about this rig as you go. You have a \
budget of {budget} tool calls and {seconds:.0f} simulated seconds for the whole trial; \
motion is interpolated at {speed:.2f} m/s, so a 0.3 m move costs about \
{cost:.0f} s of that time; descents with the gripper closed run at {lower:.2f} m/s so the \
piece does not slip in the pinch.

Embodiment notes:
- Frame: the robot base is at the origin. +x points away from the robot across the \
table, +y to the robot's left, z up; the table top is z = 0. Metres and degrees.
- The tool point is the midpoint between the fingertips. The hand always points \
straight down; `yaw` turns the fingers about the vertical axis (at yaw 0 the fingers \
close along the y axis, at yaw 90 along the x axis). The arm reaches about 0.28 to \
0.75 m from the base at low heights; targets outside the workspace box are clamped \
and reported, unreachable targets are rejected with the IK residual.
- Gripper: 1 is fully open, 0 is closed. Change the gripper in a call of its own, \
with no motion; a gripper change is given at least one second to complete. The reported \
opening is measured: fingers closed on a knob read about 0.25, which means the piece is \
held; keep commanding 0 while carrying.
- Pieces: seven 5 mm slabs, each with a 2 cm square knob, 4 cm tall, on its centroid. \
The knob is the only intended grasp. Recipe per piece: open the gripper; move to \
0.15 m above the knob centre; descend to z = {grasp:.3f} so the fingertips straddle the \
knob; close; lift to 0.15 m; carry at that height; lower to z = {release:.3f} above the \
target; open; retreat upward. The pieces are: {pieces}.
- Cameras: 'top' looks straight down on the table (image right is +x, image up is \
+y); 'wrist' looks along the tool axis with the fingertips at the bottom of the frame. \
The goal silhouette is painted on the table in both views.
- Success: all seven pieces flat on the table, inside the silhouette, each within \
about 5 mm of its place, held still. Nothing else can move the pieces."""

NUDGE = "Respond with exactly one tool call."


def tool_schemas():
    """Four tools; strict JSON schemas (every property required) for both providers."""

    def schema(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    note = {"type": "string", "description": "What you observe and why this motion."}
    gripper = {"type": "number", "description": "Gripper opening target, 0 closed to 1 open."}
    return [
        {
            "name": "move_to",
            "description": "Move the tool point to an absolute pose and set the gripper.",
            "parameters": schema(
                {
                    "x": {"type": "number", "description": "Tool point x in metres."},
                    "y": {"type": "number", "description": "Tool point y in metres."},
                    "z": {"type": "number", "description": "Tool point height in metres."},
                    "yaw": {"type": "number", "description": "Finger yaw in degrees."},
                    "gripper": gripper,
                    "note": note,
                }
            ),
        },
        {
            "name": "move_by",
            "description": "Move the tool point by a displacement and set the gripper.",
            "parameters": schema(
                {
                    "dx": {"type": "number", "description": "Displacement in x, metres."},
                    "dy": {"type": "number", "description": "Displacement in y, metres."},
                    "dz": {"type": "number", "description": "Displacement in z, metres."},
                    "dyaw": {"type": "number", "description": "Yaw change in degrees."},
                    "gripper": gripper,
                    "note": note,
                }
            ),
        },
        {
            "name": "done",
            "description": "Declare the goal achieved; the arm holds still afterwards.",
            "parameters": schema(
                {
                    "summary": {"type": "string", "description": "What was achieved."},
                    "hindsight": {
                        "type": "string",
                        "description": "What you wish you had known from the start.",
                    },
                }
            ),
        },
        {
            "name": "give_up",
            "description": "Declare the goal unachievable; the arm holds still afterwards.",
            "parameters": schema(
                {
                    "reason": {"type": "string", "description": "Why the goal is out of reach."},
                    "hindsight": {
                        "type": "string",
                        "description": "What you wish you had known from the start.",
                    },
                }
            ),
        },
    ]


def downward(yaw):
    """Hand rotation with the tool axis pointing down and the fingers turned by yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def hand_yaw(mat):
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def redacted(value):
    """A copy with image payloads replaced by their digest, for the audit log."""
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("source"), dict):
            data = value["source"].get("data", "")
            return {"type": "image", "source": {"type": "png", "sha256": digest(data)}}
        return {k: redacted(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redacted(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/png;base64,"):
        return f"data:image/png;sha256={digest(value.split(',', 1)[1])}"
    return value


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:16]


class AgentPolicy:
    """A `Policy` that lets a language model drive the arm through tool calls."""

    def __init__(self, robot, seed, config):
        if robot != "panda":
            raise ValueError("The agent adapter supports Panda only")
        from adapters import png  # Local import: adapters imports this module.

        self.png = png
        self.robot, self.seed, self.config = robot, seed, config
        self.provider = config.get("provider", "anthropic")
        if self.provider not in ("openai", "anthropic"):
            raise ValueError("Agent provider must be openai or anthropic")
        if not config.get("model"):
            raise ValueError("An explicit provider model ID is required")
        self.chunk = int(config.get("chunk_size", 25))
        self.max_calls = int(config.get("max_calls", 80))
        self.image_horizon = int(config.get("image_horizon", 2))
        self.speed = float(config.get("speed", 0.06))  # m/s of the tool point.
        self.lower_speed = float(config.get("lower_speed", LOWER_SPEED))
        self.turn_speed = np.deg2rad(float(config.get("turn_speed", 45)))  # per second.
        self.horizon_seconds = float(config.get("horizon_seconds", 300))
        self.timeout = float(config.get("timeout_seconds", 120))
        self.retries = int(config.get("retries", 3))
        self.output_tokens = int(config.get("max_output_tokens", 4096))
        self.effort = config.get("effort")
        self.use_state = bool(config.get("use_state", False))
        self.use_prompt = bool(config.get("use_prompt", True))
        self.use_images = bool(config.get("use_images", True))
        self.access = "state" if self.use_state else "pixels"
        if not (
            0 < self.timeout <= 300
            and self.chunk >= 1
            and self.max_calls >= 1
            and self.image_horizon >= 0
            and 0 < self.speed <= 0.5
            and 0 < self.lower_speed <= self.speed
            and self.turn_speed > 0
        ):
            raise ValueError("Invalid agent configuration")
        self.headers = {"Content-Type": "application/json"}
        if self.provider == "openai":
            self.url = config.get("url", OPENAI_URL)
            key = os.environ[config.get("api_key_env", "OPENAI_API_KEY")]
            self.headers["Authorization"] = "Bearer " + key
        else:
            self.url = config.get("url", ANTHROPIC_URL)
            self.headers["x-api-key"] = os.environ[config.get("api_key_env", "ANTHROPIC_API_KEY")]
            self.headers["anthropic-version"] = "2023-06-01"
        # A private model for IK, as the reference controller does.
        self.model = make_model(robot)
        self.data = mujoco.MjData(self.model)
        self.narm = len(HOME[robot])
        self.control = None
        self.history = []  # Wire-neutral turns: {"role", "content"/"items", "images"}.
        self.queue = []  # Joint actions still to hand out from the accepted move.
        self.hold = None  # Action held after done/give_up or a forced stop.
        self.gripper = None  # Last commanded opening; a pinch is a command, not a measurement.
        self.stopped = None  # {"tool", "at_step", ...} once the model ended its part.
        self.rejections = 0
        self.moves = 0
        self.records = []
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        self.last_result = None  # Tool result text shown with the next observation.
        self.subtask = ""  # The latest note; eval.py records it per tick.
        self.step = 0  # Accepted motions so far; eval.py records it per tick.

    # ----- observation -> message -----------------------------------------------

    def system_prompt(self):
        cost = 0.3 / self.speed
        return SYSTEM.format(
            budget=self.max_calls,
            seconds=self.horizon_seconds,
            speed=self.speed,
            cost=cost,
            lower=self.lower_speed,
            grasp=GRASP_HEIGHT,
            release=RELEASE_HEIGHT,
            pieces=", ".join(NAMES),
        )

    def describe(self, obs):
        step = int(round(obs["time"] / TICK))
        opening = float(np.clip(obs["qpos"][self.narm : self.narm + 2].sum() / 0.08, 0, 1))
        pos = obs["tcp_pos"]
        lines = [f"Current observation (step {step}, t = {obs['time']:.1f} s)."]
        if self.use_prompt:
            lines.append(f"Instruction: {obs['prompt']}")
        lines.append(
            f"tool: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f} m, "
            f"yaw={np.rad2deg(hand_yaw(obs['tcp_mat'])):.0f} deg, gripper={opening:.2f}"
        )
        if self.use_state:
            for name, pose in zip(NAMES, obs["pieces"]):
                yaw = np.rad2deg(2 * np.arctan2(pose[6], pose[3]))
                lines.append(
                    f"piece '{name}': x={pose[0]:.3f} y={pose[1]:.3f} "
                    f"z={pose[2] - THICKNESS / 2:.3f} m, yaw={wrap_deg(yaw):.0f} deg"
                )
            outline = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in obs["goal"])
            lines.append(f"goal outline vertices (m): {outline}")
        if self.last_result:
            lines.append(f"last tool result: {self.last_result}")
        images = []
        if self.use_images and obs.get("images"):
            for name, image in obs["images"].items():
                label = f"camera '{name}' (step {step}):"
                images.append((label, base64.b64encode(self.png(image)).decode()))
        return "\n".join(lines), images

    # ----- kinematics -----------------------------------------------------------

    def solver(self, obs):
        if self.control is None:
            proxy = SimpleNamespace(
                model=self.model,
                data=[self.data],
                narm=self.narm,
                tcp=self.model.site("tcp").id,
                limits=self.model.actuator_ctrlrange,
                observe=lambda: [obs],
            )
            self.data.qpos[: self.narm + 2] = obs["qpos"]
            mujoco.mj_forward(self.model, self.data)
            self.control = Teleop(proxy)
            self.control.posture = np.array(HOME[self.robot])
        return self.control

    def solve(self, q, position, yaw, iterations):
        """Joint configuration for a tool pose, warm-started from q; residual (m, rad)."""
        control = self.control
        self.data.qpos[: self.narm] = q
        control.position = np.asarray(position, dtype=float)
        control.orientation = downward(yaw)
        for _ in range(iterations):
            q = control.action()[: self.narm]
            self.data.qpos[: self.narm] = q
            if control.ik_error[0] < 1e-4 and control.ik_error[1] < 1e-3:
                break
        return q, control.ik_error.copy()

    def trajectory(self, obs, target, yaw, gripper):
        """Interpolate tool pose and gripper at the safe speed; IK per control tick."""
        self.solver(obs)
        start = obs["tcp_pos"].astype(float)
        start_yaw = hand_yaw(obs["tcp_mat"])
        # Interpolate the gripper from its last command, not from the measured opening:
        # fingers closed on a 2 cm knob read 0.25, and commanding 0.25 lets go of it.
        measured = float(np.clip(obs["qpos"][self.narm : self.narm + 2].sum() / 0.08, 0, 1))
        opening = measured if self.gripper is None else self.gripper
        q0 = obs["qpos"][: self.narm].astype(float)
        clamped = np.clip(target, WORKSPACE[:, 0], WORKSPACE[:, 1])
        notes = []
        for axis, before, after in zip("xyz", target, clamped):
            if abs(before - after) > 1e-9:
                notes.append(f"{axis} clamped from {before:.3f} to {after:.3f}")
        gripper = float(np.clip(gripper, 0, 1))
        turn = wrap(yaw - start_yaw)
        _, residual = self.solve(q0, clamped, start_yaw + turn, 60)
        if residual[0] > MAX_WAYPOINT_RESIDUAL or residual[1] > MAX_ANGLE_RESIDUAL:
            return None, (
                f"rejected: target unreachable (IK residual {residual[0] * 1000:.1f} mm, "
                f"{np.rad2deg(residual[1]):.1f} deg). Choose a target closer to the base, "
                "higher, or with a different yaw."
            )
        distance = float(np.linalg.norm(clamped - start))
        speed = self.speed
        if gripper < 0.5 and clamped[2] < start[2] - 1e-6:
            speed = min(speed, self.lower_speed)  # Holding and descending: no creep.
        seconds = max(distance / speed, abs(turn) / self.turn_speed, 0.2)
        if abs(gripper - opening) > 0.02:
            seconds = max(seconds, 1.0)
        count = int(math.ceil(seconds / TICK))
        actions, q, worst = [], q0, 0.0
        for k in range(1, count + 1):
            f = k / count
            q, residual = self.solve(
                q, start + f * (clamped - start), start_yaw + f * turn, 6 if k < count else 30
            )
            worst = max(worst, float(residual[0]))
            actions.append(np.r_[q, opening + f * (gripper - opening)])
        self.gripper = gripper
        notes.append(
            f"ok: moving to x={clamped[0]:.3f} y={clamped[1]:.3f} z={clamped[2]:.3f} "
            f"yaw={np.rad2deg(start_yaw + turn):.0f} deg, gripper {opening:.2f} -> "
            f"{gripper:.2f}, over {count * TICK:.1f} s ({count} steps); worst tracking "
            f"residual {worst * 1000:.1f} mm. The next observation follows the motion."
        )
        return actions, "; ".join(notes)

    # ----- tool execution -------------------------------------------------------

    def execute(self, name, args, obs):
        """Apply one tool call; returns (result text, ended?)."""
        if name in ("done", "give_up"):
            step = int(round(obs["time"] / TICK))
            self.stopped = {"tool": name, "at_step": step, **args}
            self.hold = self.queue[-1].copy() if self.queue else self.hold
            self.queue = []
            return f"{name} recorded at step {step}; the arm holds still from here.", True
        if name not in ("move_to", "move_by"):
            return f"rejected: unknown tool {name!r}.", False
        try:
            gripper = float(args["gripper"])
            if name == "move_to":
                target = np.array([float(args["x"]), float(args["y"]), float(args["z"])])
                yaw = np.deg2rad(float(args["yaw"]))
            else:
                target = obs["tcp_pos"] + np.array(
                    [float(args["dx"]), float(args["dy"]), float(args["dz"])]
                )
                yaw = hand_yaw(obs["tcp_mat"]) + np.deg2rad(float(args["dyaw"]))
            if not np.all(np.isfinite(np.r_[target, yaw, gripper])):
                raise ValueError("nonfinite")
        except (KeyError, TypeError, ValueError):
            return "rejected: every numeric argument must be a finite number.", False
        actions, result = self.trajectory(obs, target, yaw, gripper)
        if actions is None:
            return result, False
        self.queue = actions
        self.hold = actions[-1].copy()
        self.moves += 1
        self.step = self.moves
        self.subtask = str(args.get("note", ""))[:400]
        return result, False

    # ----- wire -----------------------------------------------------------------

    def request(self, tools):
        """Provider payload from the neutral history, images beyond the horizon elided."""
        with_images = [i for i, turn in enumerate(self.history) if turn.get("images")]
        keep = (
            set(with_images[len(with_images) - self.image_horizon :])
            if self.image_horizon
            else set()
        )
        if self.provider == "openai":
            items = []
            for i, turn in enumerate(self.history):
                if turn["role"] == "user":
                    parts = [{"type": "input_text", "text": turn["text"]}]
                    for label, data in turn.get("images", []):
                        parts.append({"type": "input_text", "text": label})
                        if i in keep:
                            parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{data}",
                                    "detail": "low",
                                }
                            )
                        else:
                            parts.append({"type": "input_text", "text": "[image omitted]"})
                    items.append({"role": "user", "content": parts})
                elif turn["role"] == "assistant":
                    items.extend(turn["items"])
                else:
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": turn["call_id"],
                            "output": turn["text"],
                        }
                    )
            payload = {
                "model": self.config["model"],
                "instructions": self.system_prompt(),
                "input": items,
                "tools": [{"type": "function", "strict": True, **tool} for tool in tools],
                "max_output_tokens": self.output_tokens,
                "store": False,
            }
            if self.effort:
                payload["reasoning"] = {"effort": self.effort}
                payload["include"] = ["reasoning.encrypted_content"]
            return payload
        messages = []
        for i, turn in enumerate(self.history):
            if turn["role"] == "user":
                blocks = [{"type": "text", "text": turn["text"]}]
                for label, data in turn.get("images", []):
                    blocks.append({"type": "text", "text": label})
                    if i in keep:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": data,
                                },
                            }
                        )
                    else:
                        blocks.append({"type": "text", "text": "[image omitted]"})
                messages.append({"role": "user", "content": blocks})
            elif turn["role"] == "assistant":
                messages.append({"role": "assistant", "content": turn["items"]})
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": turn["call_id"],
                                "content": turn["text"],
                            }
                        ],
                    }
                )
        payload = {
            "model": self.config["model"],
            "system": self.system_prompt(),
            "messages": messages,
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ],
            "max_tokens": self.output_tokens,
        }
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        return payload

    def post(self, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        for attempt in range(self.retries + 1):
            try:
                with urlopen(Request(self.url, body, self.headers), timeout=self.timeout) as r:
                    return json.load(r)
            except HTTPError as error:
                detail = error.read().decode(errors="replace")[:2000]
                if error.code in (408, 409, 429) or error.code >= 500:
                    if attempt < self.retries:
                        wait = error.headers.get("retry-after")
                        time.sleep(float(wait) if wait and wait.isdigit() else 2.0**attempt)
                        continue
                raise RuntimeError(f"{self.provider} HTTP {error.code}: {detail}") from None
        raise RuntimeError("unreachable")

    def parse(self, reply):
        """(assistant items to echo back, [(call_id, name, args)], text) from a reply."""
        if self.provider == "openai":
            if reply.get("status") == "incomplete":
                raise RuntimeError(f"openai incomplete: {reply.get('incomplete_details')}")
            calls, texts = [], []
            for item in reply.get("output", []):
                if item.get("type") == "function_call":
                    calls.append((item["call_id"], item["name"], json.loads(item["arguments"])))
                elif item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "refusal":
                            raise RuntimeError(f"openai refusal: {part.get('refusal')}")
                        if part.get("type") == "output_text":
                            texts.append(part["text"])
            return list(reply.get("output", [])), calls, "".join(texts)
        if reply.get("stop_reason") == "refusal":
            raise RuntimeError(f"anthropic refusal: {reply.get('stop_details')}")
        if reply.get("stop_reason") == "max_tokens":
            raise RuntimeError("anthropic output truncated; raise max_output_tokens")
        content = reply.get("content", [])
        calls = [(b["id"], b["name"], b["input"]) for b in content if b.get("type") == "tool_use"]
        texts = "".join(b["text"] for b in content if b.get("type") == "text")
        return content, calls, texts

    def call(self, tools):
        payload = self.request(tools)
        self.records.append({"request": redacted(payload)})
        self.usage["requests"] += 1
        try:
            reply = self.post(payload)
        except Exception:
            self.usage["input_tokens"] = self.usage["output_tokens"] = None
            raise
        self.records[-1]["response"] = reply
        usage = reply.get("usage", {})
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or usage.get(
            "cache_read_input_tokens", 0
        )
        for key, value in (
            ("input_tokens", usage.get("input_tokens")),
            ("output_tokens", usage.get("output_tokens")),
            ("cached_tokens", cached),
        ):
            if self.usage[key] is not None:
                self.usage[key] = None if value is None else self.usage[key] + int(value)
        return self.parse(reply)

    # ----- policy interface -----------------------------------------------------

    def act(self, obs):
        if self.queue:
            return self.slice()
        if self.stopped is not None:
            return self.hold.copy()
        if self.hold is None:
            self.hold = np.r_[obs["qpos"][: self.narm], 1.0]
        tools = tool_schemas()
        text, images = self.describe(obs)
        self.history.append({"role": "user", "text": text, "images": images})
        self.last_result = None
        while True:
            if self.usage["requests"] >= self.max_calls:
                self.stopped = {
                    "tool": "give_up",
                    "at_step": int(round(obs["time"] / TICK)),
                    "reason": "tool call budget exhausted",
                    "forced": True,
                }
                self.records.append({"forced_stop": self.stopped})
                return self.hold.copy()
            items, calls, text = self.call(tools)
            self.history.append({"role": "assistant", "items": items, "text": text})
            if not calls:
                self.rejections += 1
                if self.rejections >= MAX_REJECTIONS:
                    raise RuntimeError("Agent produced no tool call three times in a row")
                self.history.append({"role": "user", "text": NUDGE, "images": []})
                continue
            call_id, name, args = calls[0]
            result, ended = self.execute(name, args if isinstance(args, dict) else {}, obs)
            if len(calls) > 1:
                result += " Only the first tool call of a turn is executed."
            self.records[-1]["tool_result"] = {"name": name, "result": result}
            self.history.append({"role": "tool", "call_id": call_id, "text": result})
            for extra in calls[1:]:
                self.history.append(
                    {"role": "tool", "call_id": extra[0], "text": "ignored: one call per turn."}
                )
            if ended:
                return self.hold.copy()
            if result.startswith("rejected"):
                self.rejections += 1
                if self.rejections >= MAX_REJECTIONS:
                    raise RuntimeError(f"Agent had {MAX_REJECTIONS} tool calls rejected in a row")
                continue
            self.rejections = 0
            self.last_result = result
            return self.slice()

    def slice(self):
        chunk, self.queue = self.queue[: self.chunk], self.queue[self.chunk :]
        return np.asarray(chunk, dtype=float)

    def audit(self):
        return {
            "usage": self.usage,
            "moves": self.moves,
            "stopped": self.stopped,
            "transcript": self.records,
        }


def wrap_deg(angle):
    return (angle + 180) % 360 - 180
