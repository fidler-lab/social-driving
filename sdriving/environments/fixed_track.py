import math
import random
from collections import deque
from itertools import product

import numpy as np
import torch

from gym.spaces import Box, Discrete, Tuple
from sdriving.environments.intersection import (
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment,
)
from sdriving.tsim import (
    generate_intersection_world_4signals,
    BicycleKinematicsModel,
    angle_normalize,
    BatchedVehicle,
    World,
    intervehicle_collision_check,
)


class MultiAgentRoadIntersectionFixedTrackEnvironment(
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment
):
    def __init__(self, *args, **kwargs):
        self.lane_side = 1
        super().__init__(*args, **kwargs)

    def get_action_space(self):
        self.max_accln = 1.5
        self.normalization_factor = torch.as_tensor([self.max_accln, 1.0])
        return Box(
            low=np.array([-self.max_accln]), high=np.array([self.max_accln]),
        )

    def discrete_to_continuous_actions(self, action: torch.Tensor):
        return torch.cat([torch.zeros_like(action), action], dim=-1)


class MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment(
    MultiAgentRoadIntersectionFixedTrackEnvironment
):
    def configure_action_space(self):
        self.max_accln = 1.5
        self.action_list = torch.arange(
            -self.max_accln, self.max_accln, step=0.25
        ).unsqueeze(1)
        self.action_list = torch.cat(
            [torch.zeros_like(self.action_list), self.action_list], dim=-1
        )

    def get_action_space(self):
        self.normalization_factor = torch.as_tensor([self.max_accln, 1.0])
        return Discrete(self.action_list.size(0))

    def discrete_to_continuous_actions(self, action: torch.Tensor):
        return self.action_list[action]
