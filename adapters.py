"""Local checkpoint, HTTP, OpenAI Responses and Anthropic Messages policy adapters.

The provider adapters are the language-model track: every inference call sends
the prompt, the state (unless `use_state` is false) and, with `--obs pixels`, the
top and wrist cameras as PNG images, and asks for a JSON chunk of joint actions
through each provider's structured-output feature. Keys and model IDs come from
the environment or from a `.env` file at the repository root, never from the
system file. Requests, responses and token usage are logged beside the
trajectory; the key is not. No provider SDK is required: both APIs are plain
HTTPS with JSON, so the whole contract is visible in this file.
"""

import base64
import importlib.util
import json
import os
import struct
import time
import zlib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from env import make_model

ROOT = Path(__file__).resolve().parent
OPENAI_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
    "required": ["actions"],
    "additionalProperties": False,
}


def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines into the environment; values already set win."""
    path = Path(path)
    if not path.is_file():
        return {}
    loaded = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_module(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy


def system_config(path):
    """Resolve declared local paths before hashing and constructing a system."""
    path = Path(path)
    config = json.loads(path.read_text())
    if config.get("type") not in ("local", "http", "openai", "anthropic"):
        raise ValueError("System type must be local, http, openai or anthropic")
    if "api_key" in config:
        raise ValueError("Use api_key_env, not a literal API key in the system file")
    config.setdefault("kwargs", {})
    load_env()
    if "model_env" in config:
        config["model"] = os.environ[config["model_env"]]
    files = [path.resolve()]
    for field in ("module", "checkpoint"):
        if field in config:
            resolved = (path.parent / config[field]).resolve()
            config[field] = str(resolved)
            files.append(resolved)
    for artifact in config.get("artifacts", []):
        files.append((path.parent / artifact).resolve())
    return config, files


def policy_factory(config):
    if config["type"] == "local":
        cls = load_module(config["module"])
        kwargs = dict(config.get("kwargs", {}))
        if "checkpoint" in config:
            kwargs["checkpoint"] = config["checkpoint"]
        return lambda robot, seed: cls(robot=robot, seed=seed, **kwargs)
    return lambda robot, seed: RemotePolicy(robot, seed, config)


def serializable(obs, decimals=4):
    """JSON-ready observation: arrays rounded to 0.1 mm / 0.1 mrad, images dropped."""
    out = {}
    for key, value in obs.items():
        if key == "images":
            continue
        out[key] = np.round(value, decimals).tolist() if isinstance(value, np.ndarray) else value
    return out


def png(image):
    """Encode an (H, W, 3) uint8 array as PNG; pure Python, no image library."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, _ = image.shape
    rows = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 6))
        + chunk(b"IEND", b"")
    )


def encoded_images(obs):
    """Ordered (name, base64 PNG) pairs, or an empty list without pixels."""
    images = obs.get("images") or {}
    return [(name, base64.b64encode(png(images[name])).decode()) for name in images]


class RemotePolicy:
    def __init__(self, robot, seed, config):
        self.robot, self.seed, self.config = robot, seed, config
        self.kind = config["type"]
        self.records = []
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        self.timeout = float(config.get("timeout_seconds", 60))
        self.chunk = int(config.get("chunk_size", 1))
        self.output_tokens = int(config.get("max_output_tokens", 4096))
        self.retries = int(config.get("retries", 3))
        self.effort = config.get("effort")
        self.use_state = bool(config.get("use_state", True))
        self.use_images = bool(config.get("use_images", True))
        self.access = "state" if self.use_state else "pixels"
        if not 0 < self.timeout <= 300 or self.chunk < 1 or self.output_tokens < 1:
            raise ValueError("Invalid timeout, chunk size or output-token limit")
        model = make_model(robot)
        self.narm = model.nu - 1
        self.limits = model.actuator_ctrlrange[: self.narm].tolist()
        self.headers = {"Content-Type": "application/json"}
        if self.kind == "http":
            self.url = config["url"]
            key_env = config.get("api_key_env")
            if key_env:
                self.headers["Authorization"] = "Bearer " + os.environ[key_env]
        else:
            if not config.get("model"):
                raise ValueError("An explicit provider model ID is required")
            if self.kind == "openai":
                self.url = config.get("url", OPENAI_URL)
                self.headers["Authorization"] = (
                    "Bearer " + os.environ[config.get("api_key_env", "OPENAI_API_KEY")]
                )
            else:
                self.url = config.get("url", ANTHROPIC_URL)
                self.headers["x-api-key"] = os.environ[
                    config.get("api_key_env", "ANTHROPIC_API_KEY")
                ]
                self.headers["anthropic-version"] = "2023-06-01"

    def instructions(self):
        return (
            f"You control a {self.robot} robot arm in a MuJoCo simulation that must assemble "
            "a seven-piece tangram into the silhouette painted on the table. "
            f"Reply with JSON only: an object with an 'actions' array of 1 to {self.chunk} "
            f"commands. Each command is {self.narm} absolute arm joint positions in radians "
            "followed by the gripper opening from 0 (closed) to 1 (open). "
            f"Arm joint limits in order: {self.limits}. Commands execute open loop at 50 Hz, "
            "one per 0.02 s, and each joint target may move at most 0.04 rad per command. "
            "World coordinates are metres and quaternions are wxyz. Every piece has a 2 cm "
            "square knob on its top face: grasp the knob from above, lift, carry, lower and "
            "release. Image 1 is the top camera and image 2 the wrist camera when present. "
            "Use the observation you are given; nothing else can move the pieces."
        )

    def content(self, obs):
        """Observation text plus labelled images, images first, for either provider."""
        observation = serializable(obs)
        if not self.config.get("use_prompt", True):
            observation.pop("prompt", None)
        if not self.use_state:
            for key in ("qpos", "qvel", "tcp_pos", "tcp_mat", "pieces", "piece_velocities", "goal"):
                observation.pop(key, None)
        images = encoded_images(obs) if self.use_images else []
        text = "Observation:\n" + json.dumps(observation, allow_nan=False)
        return observation, images, text

    def payload(self, obs):
        observation, images, text = self.content(obs)
        if self.kind == "http":
            return {
                "robot": self.robot,
                "seed": self.seed,
                "observation": observation,
                "images": {name: data for name, data in images},
                "action_contract": {
                    "type": "joint_position",
                    "arm_limits": self.limits,
                    "gripper": [0, 1],
                    "max_chunk": self.chunk,
                },
                "config": self.config.get("kwargs", {}),
            }
        if self.kind == "openai":
            parts = []
            for i, (name, data) in enumerate(images, start=1):
                parts.append({"type": "input_text", "text": f"Image {i}: {name} camera"})
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{data}",
                        "detail": "low",
                    }
                )
            parts.append({"type": "input_text", "text": text})
            payload = {
                "model": self.config["model"],
                "instructions": self.instructions(),
                "input": [{"role": "user", "content": parts}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "actions",
                        "strict": True,
                        "schema": ACTION_SCHEMA,
                    }
                },
                "max_output_tokens": self.output_tokens,
                "store": False,
            }
            if self.effort:
                payload["reasoning"] = {"effort": self.effort}
            return payload
        blocks = []
        for i, (name, data) in enumerate(images, start=1):
            blocks.append({"type": "text", "text": f"Image {i}: {name} camera"})
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": data},
                }
            )
        blocks.append({"type": "text", "text": text})
        payload = {
            "model": self.config["model"],
            "system": self.instructions(),
            "messages": [{"role": "user", "content": blocks}],
            "max_tokens": self.output_tokens,
            "output_config": {"format": {"type": "json_schema", "schema": ACTION_SCHEMA}},
        }
        if self.effort:
            payload["output_config"]["effort"] = self.effort
        return payload

    def post(self, payload):
        """One request with bounded retries on rate limits and server errors."""
        body = json.dumps(payload, allow_nan=False).encode()
        for attempt in range(self.retries + 1):
            request = Request(self.url, body, self.headers)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as error:
                detail = error.read().decode(errors="replace")[:2000]
                if error.code in (408, 409, 429) or error.code >= 500:
                    if attempt < self.retries:
                        wait = error.headers.get("retry-after")
                        time.sleep(float(wait) if wait and wait.isdigit() else 2.0**attempt)
                        continue
                raise RuntimeError(f"{self.kind} HTTP {error.code}: {detail}") from None
        raise RuntimeError("unreachable")

    def parse(self, reply):
        """Provider-specific unwrapping to the JSON text, refusing truncated output."""
        if self.kind == "http":
            return reply
        if self.kind == "openai":
            if reply.get("status") == "incomplete":
                raise RuntimeError(f"openai incomplete: {reply.get('incomplete_details')}")
            texts = []
            for item in reply.get("output", []):
                if item.get("type") != "message":
                    continue
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        raise RuntimeError(f"openai refusal: {part.get('refusal')}")
                    if part.get("type") == "output_text":
                        texts.append(part["text"])
            return json.loads("".join(texts))
        if reply.get("stop_reason") == "refusal":
            raise RuntimeError(f"anthropic refusal: {reply.get('stop_details')}")
        if reply.get("stop_reason") == "max_tokens":
            raise RuntimeError("anthropic output truncated; raise max_output_tokens")
        return json.loads(
            "".join(b["text"] for b in reply.get("content", []) if b.get("type") == "text")
        )

    def act(self, obs):
        payload = self.payload(obs)
        record = {"request": payload}
        self.records.append(record)
        self.usage["requests"] += 1
        try:
            reply = self.post(payload)
        except Exception:
            self.usage["input_tokens"] = self.usage["output_tokens"] = None
            raise
        record["response"] = reply
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
        actions = np.asarray(self.parse(reply)["actions"], dtype=float)
        if actions.ndim != 2 or not 1 <= len(actions) <= self.chunk:
            raise ValueError("Provider must return a nonempty bounded actions array")
        return actions

    def audit(self):
        return {"usage": self.usage, "transcript": self.records}
