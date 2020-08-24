import math
import random
from collections import deque
from itertools import product

import numpy as np
import torch

from gym.spaces import Box, Discrete, Tuple
from sdriving.environments.intersection import (
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment
)
from sdriving.tsim import (
    SplineModel,
    angle_normalize,
    BatchedVehicle,
    intervehicle_collision_check,
)
from sdriving.nuscenes import NuscenesWorld


class MultiAgentNuscenesIntersectionDrivingEnvironment(
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment
):
    def __init__(
        self,
        map_path: str,  # For now test with a single map
        npoints: int = 360,
        horizon: int = 300,
        timesteps: int = 10,
        history_len: int = 5,
        time_green: int = 100,
        nagents: int = 4,
        device: torch.device = torch.device("cpu"),
        lidar_noise: float = 0.0,
    ):
        self.npoints = npoints
        self.history_len = history_len
        self.time_green = time_green
        self.device = device

        world = NuscenesWorld(map_path)
        super(
            MultiAgentRoadIntersectionBicycleKinematicsEnvironment, self
        ).__init__(world, nagents, horizon, timesteps, device)
        
        self.queue1 = None
        self.queue2 = None
        self.lidar_noise = lidar_noise
        
        bool_buffer = torch.ones(self.nagents * 4, self.nagents * 4)
        for i in range(0, self.nagents * 4, 4):
            bool_buffer[i : (i + 4), i : (i + 4)] -= 1
        self.bool_buffer = bool_buffer.bool()
        
        self.buffered_ones = torch.ones(self.nagents, 1, device=self.device)
    
    def get_state(self):
        a_id = self.get_agent_ids_list()[0]
        ts = self.world.get_all_traffic_signal().unsqueeze(1)
        vehicle = self.agents[a_id]
        head = vehicle.orientation
        
        dist = vehicle.distance_from_destination()
        path_distance = self.dynamics.distance_proxy - self.dynamics.distances
        inv_dist = torch.where(dist == 0, self.buffered_ones, 1 / path_distance)
        
        speed = vehicle.speed
        
        obs = torch.cat([ts, speed / self.dynamics.v_lim, head, inv_dist], -1)
        lidar = 1 / self.world.get_lidar_data_all_vehicles(self.npoints)

        if self.lidar_noise > 0:
            lidar *= torch.rand_like(lidar) > self.lidar_noise

        if self.history_len > 1:
            while len(self.queue1) <= self.history_len - 1:
                self.queue1.append(obs)
                self.queue2.append(lidar)
            self.queue1.append(obs)
            self.queue2.append(lidar)

            return (
                torch.cat(list(self.queue1), dim=-1),
                torch.cat(list(self.queue2), dim=-1),
            )
        else:
            return obs, lidar
        
    def get_reward(self, new_collisions: torch.Tensor, action: torch.Tensor):
        a_ids = self.get_agent_ids_list()
        a_id = a_ids[0]
        vehicle = self.agents[a_id]

        # Distance from destination
        # A bit hacky, but doesn't matter if the agent only goes forward
        distances = self.dynamics.distance_proxy - self.dynamics.distances

        # Agent Speeds
        speeds = vehicle.speed

        # Action Regularization
        if self.cached_actions is not None:
            smoothness = (
                (action - self.cached_actions).pow(2)
                / (2 * self.normalization_factor).pow(2)
            ).sum(-1, keepdim=True)
        else:
            smoothness = 0.0

        # Goal Reach Bonus
        reached_goal = distances <= 10.0
        not_completed = ~self.completion_vector
        goal_reach_bonus = (not_completed * reached_goal).float()
        self.completion_vector = self.completion_vector + reached_goal
        for v in a_ids:
            self.agents[v].destination = self.agents[
                v
            ].position * self.completion_vector + self.agents[
                v
            ].destination * (
                ~self.completion_vector
            )

        distances *= not_completed / self.dynamics.distance_proxy

        # Collision
        new_collisions = ~self.collision_vector * new_collisions
        penalty = (
            new_collisions.float()
            + new_collisions
            * distances
            * (self.horizon - self.nsteps - 1)
            / self.horizon
        )
        
        self.collision_vector += new_collisions

        return (
            -(distances + smoothness) * (~self.collision_vector) / self.horizon
            - (speeds / 8.0).abs() * self.completion_vector / self.horizon
            - penalty
            + goal_reach_bonus
        )
    
    def add_vehicles_to_world(self):
        vehicle = None
        dims = torch.as_tensor([[4.48, 2.2]])
        self.cps = []
        for _ in range(self.nagents):
            successful_placement = False
            while not successful_placement:
                idx, (spos, epos, orient, dorient, cps) = self.world.sample_new_vehicle_position()
                if vehicle is None:
                    vehicle = BatchedVehicle(
                        position=spos,
                        orientation=orient,
                        destination=epos,
                        dest_orientation=dorient,
                        dimensions=dims,
                        initial_speed=torch.zeros(1, 1),
                        name="agent",
                    )
                    break
                else:
                    successful_placement = vehicle.add_vehicle(
                        position=spos,
                        orientation=orient,
                        destination=epos,
                        dest_orientation=dorient,
                        dimensions=dims,
                        initial_speed=torch.zeros(1, 1),
                    )
            self.cps.append(cps)
        self.cps = torch.cat(self.cps)

        vehicle.add_bool_buffer(self.bool_buffer)

        # TODO: Send the sampled idxs
        self.world.add_vehicle(vehicle, None)
        self.store_dynamics(vehicle)
        self.agents[vehicle.name] = vehicle

        self.original_distances = vehicle.distance_from_destination()

    def store_dynamics(self, vehicle):
        self.dynamics = SplineModel(
            self.cps, v_lim=torch.ones(self.nagents) * 8.0
        )

    def reset(self):
        # Keep the environment fixed for now
        self.world.reset()
        self.add_vehicles_to_world()

        self.queue1 = deque(maxlen=self.history_len)
        self.queue2 = deque(maxlen=self.history_len)

        return super(MultiAgentRoadIntersectionBicycleKinematicsEnvironment, self).reset()