"""Run a LeRobot checkpoint (SmolVLA, ACT, pi0.5, ...) as a Tangram-Bench policy.

System file, paths relative to the JSON:
  {"type": "local", "module": "../lerobot_policy.py",
   "checkpoint": "../../outputs/train/smolvla/checkpoints/last/pretrained_model",
   "kwargs": {"fps": 10, "n_action_steps": 10}}

Evaluate with pixels and a chunk budget of n_action_steps * 50 / fps:
  uv run eval.py --system examples/systems/lerobot.json --obs pixels --max-chunk 50 \\
      --steps 15000 --max-inference-calls 15000 --out outputs/runs/smolvla-dev.json

Observations map to the dataset written by tools/export.py: top and wrist
images, `qpos` as observation.state, the prompt as the task. The policy
predicts one chunk per call at the dataset frame rate; each action is repeated
50 / fps control ticks, which is how the demonstrations were recorded. The
checkpoint is loaded once per process and reused across episodes. The SmolVLM2
backbone and tokenizer come from the Hub cache, the checkpoints/ folder by default.
Requires the `lerobot` extra: uv sync --extra lerobot
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", str(Path(__file__).resolve().parents[1] / "checkpoints"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env import CAMERAS, make_model  # noqa: E402

_CACHE = {}


def load(checkpoint, device):
    """Load policy and its pre/post processors once; later episodes reuse them."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    key = (str(checkpoint), device)
    if key not in _CACHE:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
        policy = policy.to(device).eval()
        pre, post = make_pre_post_processors(
            config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        _CACHE[key] = (policy, pre, post)
    return _CACHE[key]


class Policy:
    access = "pixels"

    def __init__(
        self, robot, seed, checkpoint, fps=10, n_action_steps=None, device=None, task=None
    ):
        if 50 % fps:
            raise ValueError("fps must divide the 50 Hz control rate")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy, self.pre, self.post = load(checkpoint, device)
        self.policy.reset()
        self.steps = int(n_action_steps or self.policy.config.n_action_steps)
        if not 1 <= self.steps <= self.policy.config.chunk_size:
            raise ValueError("n_action_steps must be within the policy chunk size")
        self.repeat = 50 // fps
        self.task = task
        model = make_model(robot)
        self.narm = model.nu - 1
        self.limits = model.actuator_ctrlrange[: self.narm]
        torch.manual_seed(seed)

    def act(self, obs):
        if "images" not in obs:
            raise ValueError("This policy needs pixels; run eval.py with --obs pixels")
        batch = {
            f"observation.images.{name}": torch.from_numpy(
                np.ascontiguousarray(obs["images"][name])
            )
            .permute(2, 0, 1)
            .float()
            / 255
            for name in CAMERAS
        }
        batch["observation.state"] = torch.from_numpy(np.asarray(obs["qpos"], dtype=np.float32))
        batch["task"] = self.task if self.task is not None else obs["prompt"]
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(batch))
            chunk = self.post(chunk)
        actions = np.asarray(chunk.detach().cpu().numpy(), dtype=float)
        actions = actions.reshape(-1, actions.shape[-1])[: self.steps, : self.narm + 1]
        actions[:, :-1] = np.clip(actions[:, :-1], self.limits[:, 0], self.limits[:, 1])
        actions[:, -1] = np.clip(actions[:, -1], 0, 1)
        return np.repeat(actions, self.repeat, axis=0)
