"""Edit this file to try a policy. No inheritance, registry, or training dependency.

Contract: Policy(robot=..., seed=...).act(observation) -> n_arm joint targets
in radians, followed by gripper opening in [0,1]. A new instance is made per episode.
This baseline holds the starting arm pose and should score zero success.
"""

import numpy as np


class Policy:
    def __init__(self, robot, seed):
        self.narm = 7 if robot == "panda" else 6
        self.target = None

    def act(self, obs):
        if self.target is None:
            self.target = np.r_[obs["qpos"][: self.narm], 1.0]
        return self.target.copy()
