import itertools
import math
import random
from collections import deque

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple
from sdriving.envs.intersection_env import RoadIntersectionContinuousControlEnv
from sdriving.trafficsim.common_networks import (
    generate_intersection_world_4signals,
    generate_intersection_world_12signals,
)
from sdriving.trafficsim.dynamics import BicycleKinematicsModel
from sdriving.trafficsim.utils import (
    angle_normalize,
    circle_segment_area 
)
from sdriving.trafficsim.vehicle import Vehicle
from sdriving.trafficsim.world import World


class DifferentiableRoadIntersectionControlEnv(
    RoadIntersectionContinuousControlEnv
):
    def generate_world_without_agents(self):
        self.max_length = 70.0
        self.max_width = 30.0
        self.length = 50.0 # (torch.rand(1) * 30.0 + 40.0).item()
        self.width = 15.0 # (torch.rand(1) * 15.0 + 15.0).item()
        time_green = int((torch.rand(1) / 2 + 1) * self.time_green)
        return generate_intersection_world_4signals(
            length=self.length,
            road_width=self.width,
            name="traffic_signal_world",
            time_green=time_green,
            ordering=random.choice([0, 1]),
        )

    def _get_distances(
        self, pt: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor
    ):
        if pt.ndim == 1:
            pt = pt.unsqueeze(0)
        l2 = (pt1 - pt2).pow(2).sum(1)
        t = ((pt - pt1).matmul((pt2 - pt1).T)).diag() / l2
        t = torch.max(
            torch.zeros_like(t), torch.min(torch.ones_like(t), t)
        ).unsqueeze(1)
        projection = pt1 + t * (pt2 - pt1)
        distance = (pt - projection).pow(2).sum(1).sqrt()
        return distance

    def smooth_collision_penalty(self, a_id: str, pos=None):
        vehicle = self.agents[a_id]["vehicle"]
        ventity = self.world.vehicles[a_id]
        pt1, pt2 = self.world.road_network.get_neighbouring_edges(
            ventity.road,
            vname=a_id,
            type="road" if not ventity.grayarea else "garea",
            cars=False,
        )
        distance = self._get_distances(
            vehicle.position if pos is None else pos, pt1, pt2
        )

        return circle_segment_area(distance, vehicle.safety_circle)

    def get_reward(self):
        rewards = {}
        for a_id in self.get_agent_ids_list():
            distance = self.agents[a_id]["vehicle"].distance_from_point(
                self.agents[a_id]["original_destination"]
            )
            collision = (
                torch.mean(self.smooth_collision_penalty(a_id))
                / self.agents[a_id]["vehicle"].area
            )
            # if self.world.check_collision(a_id):
            #     self.agents[a_id]["done"] = True

            rewards[a_id] = (
                1.0 * distance / self.agents[a_id]["original_distance"]
                # + 10000.0 * collision
            ) / self.horizon
        return rewards
