"""Explicit local, HTTP, OpenAI Responses and Anthropic Messages policy adapters.

Provider model IDs are supplied by the user. No SDK or credentials are required
for local policies. Remote adapters log payloads/responses, never auth headers.
"""

import importlib.util
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from env import make_model


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


def serializable(obs):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in obs.items()
    }


class RemotePolicy:
    def __init__(self, robot, seed, config):
        self.robot, self.seed, self.config = robot, seed, config
        self.kind = config["type"]
        self.records = []
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self.timeout = float(config.get("timeout_seconds", 60))
        self.chunk = int(config.get("chunk_size", 1))
        self.output_tokens = int(config.get("max_output_tokens", 2048))
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
                self.url = config.get("url", "https://api.openai.com/v1/responses")
                self.headers["Authorization"] = (
                    "Bearer " + os.environ[config.get("api_key_env", "OPENAI_API_KEY")]
                )
            else:
                self.url = config.get("url", "https://api.anthropic.com/v1/messages")
                self.headers["x-api-key"] = os.environ[
                    config.get("api_key_env", "ANTHROPIC_API_KEY")
                ]
                self.headers["anthropic-version"] = "2023-06-01"

    def act(self, obs):
        observation = serializable(obs)
        if not self.config.get("use_prompt", True):
            observation.pop("prompt", None)
        instructions = (
            f"Return only JSON with an actions array of 1..{self.chunk} commands. "
            f"Each command has {self.narm} absolute arm joint positions in radians, "
            "followed by gripper opening from 0 (closed) to 1 (open). "
            f"Arm limits: {self.limits}. World coordinates are metres, quaternions wxyz. "
            "Commands execute every 0.02 simulated seconds with a 0.04 radian slew limit. "
            "Use the goal polygon and current observed state. No direct piece motion is available."
        )
        content = json.dumps(observation, allow_nan=False)
        if self.kind == "http":
            payload = {
                "robot": self.robot,
                "seed": self.seed,
                "observation": observation,
                "action_contract": {
                    "type": "joint_position",
                    "arm_limits": self.limits,
                    "gripper": [0, 1],
                    "max_chunk": self.chunk,
                },
                "config": self.config.get("kwargs", {}),
            }
        elif self.kind == "openai":
            payload = {
                "model": self.config["model"],
                "instructions": instructions,
                "input": content,
                "max_output_tokens": self.output_tokens,
                "store": False,
            }
        else:
            payload = {
                "model": self.config["model"],
                "system": instructions,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.output_tokens,
            }
        record = {"request": payload}
        self.records.append(record)
        self.usage["requests"] += 1
        request = Request(self.url, json.dumps(payload, allow_nan=False).encode(), self.headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                reply = json.load(response)
        except Exception:
            self.usage["input_tokens"] = self.usage["output_tokens"] = None
            raise
        record["response"] = reply
        usage = reply.get("usage", {})
        for key in ("input_tokens", "output_tokens"):
            self.usage[key] = (
                None
                if self.usage[key] is None or key not in usage
                else self.usage[key] + int(usage[key])
            )
        if self.kind == "openai":
            texts = [
                c["text"]
                for item in reply.get("output", [])
                if item.get("type") == "message"
                for c in item.get("content", [])
                if c.get("type") == "output_text"
            ]
            result = json.loads("".join(texts))
        elif self.kind == "anthropic":
            result = json.loads(
                "".join(c["text"] for c in reply.get("content", []) if c.get("type") == "text")
            )
        else:
            result = reply
        actions = np.asarray(result["actions"], dtype=float)
        if actions.ndim != 2 or not 1 <= len(actions) <= self.chunk:
            raise ValueError("Provider must return a nonempty bounded actions array")
        return actions

    def audit(self):
        return {"usage": self.usage, "transcript": self.records}
