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
    generate_intersection_world_12signals,
    FixedTrackAccelerationModel,
    angle_normalize,
    BatchedVehicle,
    World,
    intervehicle_collision_check,
)


class MultiAgentRoadIntersectionFixedTrackEnvironment(
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment
):
    def __init__(
        self, *args, turns: bool = False, default_color: bool = True, **kwargs
    ):
        self.lane_side = 1
        self.turns = turns
        self.default_color = default_color
        super().__init__(*args, **kwargs)

    def generate_world_without_agents(self):
        if not self.turns:
            return super().generate_world_without_agents()
        length = (torch.rand(1) * 25.0 + 55.0).item()
        width = (torch.rand(1) * 15.0 + 15.0).item()
        time_green = int((torch.rand(1) / 2 + 1) * self.time_green)
        return (
            generate_intersection_world_12signals(
                length=length,
                road_width=width,
                name="traffic_signal_world",
                time_green=time_green,
                ordering=random.choice(range(8)),
                default_colmap=self.default_color,
            ),
            {"length": length, "width": width},
        )

    def store_dynamics(self, vehicle):
        if not self.turns:
            return super().store_dynamics(vehicle)

        w2 = self.width / 2
        center = torch.as_tensor([[w2, w2], [-w2, w2], [-w2, -w2], [w2, -w2]])

        centers, radii, distances = [], [], []
        for i in range(vehicle.nbatch):
            srd, erd = self.srd[i], self.erd[i]
            pos = vehicle.position[i].abs()
            l, m = (srd + 1) % 2, srd % 2
            p = pos[l : (l + 1)]
            if erd == (srd + 1) % 4:
                r = w2 + p
                c = center[srd : (srd + 1), :]
            elif erd == (srd + 2) % 4:
                r = torch.zeros(1, device=self.device)
                c = center[srd : (srd + 1), :]  # Just a mock point
            else:
                r = w2 - p
                c = center[erd : (erd + 1), :]
            d = pos[m : (m + 1)] - w2
            centers.append(c)
            radii.append(r)
            distances.append(d)

        self.dynamics = FixedTrackAccelerationModel(
            theta1=vehicle.orientation[:, 0],
            theta2=vehicle.dest_orientation[:, 0],
            radius=torch.cat(radii),
            center=torch.cat(centers),
            distance1=torch.cat(distances),
            v_lim=torch.ones(self.nagents) * 8.0,
        )

    def get_action_space(self):
        self.max_accln = 1.5
        if self.turns:
            self.normalization_factor = torch.as_tensor([self.max_accln])
        else:
            self.normalization_factor = torch.as_tensor([1.0, self.max_accln])
        return Box(
            low=np.array([-self.max_accln]),
            high=np.array([self.max_accln]),
        )

    def discrete_to_continuous_actions(self, action: torch.Tensor):
        if self.turns:
            return action
        return torch.cat([torch.zeros_like(action), action], dim=-1)


class MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment(
    MultiAgentRoadIntersectionFixedTrackEnvironment
):
    def configure_action_space(self):
        self.max_accln = 1.5
        self.action_list = torch.arange(
            -self.max_accln, self.max_accln + 0.05, step=0.25
        ).unsqueeze(1)
        if not self.turns:
            self.action_list = torch.cat(
                [torch.zeros_like(self.action_list), self.action_list], dim=-1
            )

    def get_action_space(self):
        if self.turns:
            self.normalization_factor = torch.as_tensor([self.max_accln])
        else:
            self.normalization_factor = torch.as_tensor([1.0, self.max_accln])
        return Discrete(self.action_list.size(0))

    def discrete_to_continuous_actions(self, action: torch.Tensor):
        return self.action_list[action]
