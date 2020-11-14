import math
import random
from collections import deque
from itertools import product

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple

from sdriving.environments.base_env import BaseMultiAgentDrivingEnvironment
from sdriving.tsim import (
    BatchedVehicle,
    BicycleKinematicsModel,
    World,
    angle_normalize,
    generate_intersection_world_4signals,
    intervehicle_collision_check,
)


class MultiAgentRoadIntersectionBicycleKinematicsEnvironment(
    BaseMultiAgentDrivingEnvironment
):
    def __init__(
        self,
        npoints: int = 360,
        horizon: int = 200,
        timesteps: int = 25,
        history_len: int = 5,
        time_green: int = 100,
        nagents: int = 4,
        device: torch.device = torch.device("cpu"),
        balance_cars: bool = True,
        mode: int = 1,
        lidar_noise: float = 0.0,
        vision_range: float = 30.0,
    ):
        self.npoints = npoints
        self.history_len = history_len
        self.time_green = time_green
        self.device = device
        world, config = self.generate_world_without_agents()
        for k, v in config.items():
            setattr(self, k, v)
        super().__init__(world, nagents, horizon, timesteps, device)
        self.queue1 = None
        self.queue2 = None
        self.lidar_noise = lidar_noise
        self.mode = mode
        self.balance_cars = balance_cars
        self.vision_range = vision_range

        bool_buffer = torch.ones(self.nagents * 4, self.nagents * 4)
        for i in range(0, self.nagents * 4, 4):
            bool_buffer[i : (i + 4), i : (i + 4)] -= 1
        self.bool_buffer = bool_buffer.bool()

    def generate_world_without_agents(self):
        length = (torch.rand(1) * 25.0 + 55.0).item()
        width = (torch.rand(1) * 15.0 + 15.0).item()
        time_green = int((torch.rand(1) / 2 + 1) * self.time_green)
        return (
            generate_intersection_world_4signals(
                length=length,
                road_width=width,
                name="traffic_signal_world",
                time_green=time_green,
                ordering=random.choice([0, 1]),
            ),
            {"length": length, "width": width},
        )

    def get_observation_space(self):
        return Tuple(
            [
                Box(
                    low=np.array(
                        [0.0, -1.0, -math.pi, 0.0] * self.history_len
                    ),
                    high=np.array(
                        [1.0, 1.0, math.pi, np.inf] * self.history_len
                    ),
                ),
                Box(0.0, np.inf, shape=(self.npoints * self.history_len,)),
            ]
        )

    def get_action_space(self):
        self.max_accln = 1.5
        self.max_steering = 0.1
        self.normalization_factor = torch.as_tensor(
            [self.max_steering, self.max_accln]
        )
        return Box(
            low=np.array([-self.max_steering, -self.max_accln]),
            high=np.array([self.max_steering, self.max_accln]),
        )

    def get_state(self):
        a_ids = self.get_agent_ids_list()
        ts = self.world.get_all_traffic_signal().unsqueeze(1)
        head = torch.cat([self.agents[v].optimal_heading() for v in a_ids])
        dist = torch.cat(
            [self.agents[v].distance_from_destination() for v in a_ids]
        )
        inv_dist = 1 / dist.clamp(min=1.0)
        speed = torch.cat([self.agents[v].speed for v in a_ids])

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
                (
                    torch.cat(list(self.queue1), dim=-1),
                    torch.cat(list(self.queue2), dim=-1),
                ),
                self.agent_names,
            )
        else:
            return ((obs, lidar), self.agent_names)

    def get_reward(self, new_collisions: torch.Tensor, action: torch.Tensor):
        a_ids = self.get_agent_ids_list()

        # Distance from destination
        # FIXME: Taking L2 distance creates issues in BKM
        distances = torch.cat(
            [self.agents[v].distance_from_destination() for v in a_ids]
        )

        # Agent Speeds
        speeds = torch.cat([self.agents[v].speed for v in a_ids])

        # Action Regularization
        if self.cached_actions is not None:
            smoothness = (
                (action - self.cached_actions).pow(2)
                / (2 * self.normalization_factor).pow(2)
            ).sum(-1, keepdim=True)
        else:
            smoothness = 0.0

        # Goal Reach Bonus
        reached_goal = distances <= self.width / 3
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

        distances *= not_completed / self.original_distances

        # Collision
        new_collisions = ~self.collision_vector * new_collisions
        self.agents["agent"].speed *= ~new_collisions
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

    def _sample_vehicle_on_road(self, srd: int, erd: int):
        sroad = self.world.road_network.roads[f"traffic_signal_world_{srd}"]
        eroad = self.world.road_network.roads[f"traffic_signal_world_{erd}"]

        orientation = angle_normalize(
            sroad.orientation.unsqueeze(0) + math.pi
        ).clone()
        dest_orientation = eroad.orientation.unsqueeze(0).clone()

        end_pos = torch.zeros(1, 2)
        end_pos[0, erd % 2] = (
            -0.5 * (self.length * 0.67 + self.width) * np.sign(erd - 1.5)
        )
        str_pos = sroad.sample(x_bound=0.6, y_bound=0.6)
        if hasattr(self, "lane_side"):
            side = self.lane_side * np.sign(srd % 3 - 0.5)
            str_pos[0, (srd + 1) % 2] = (
                side * (torch.rand(1) * 0.15 + 0.15) * self.width
            )

        # The sampling guarantees no collision with road boundaries
        return str_pos, end_pos, orientation, dest_orientation

    def add_vehicles_to_world(self):
        # For now let us not do any turns
        vehicle = None
        dims = torch.as_tensor([[4.48, 2.2]])

        # TODO: check for intervehicle collision while generating environments
        #       for <= 4 agents, there is no need to check for collisions
        srd = np.random.choice([0, 1, 2, 3])
        self.srd = []
        self.erd = []
        for _ in range(self.actual_nagents):

            successful_placement = False
            while not successful_placement:
                srd_new = (
                    (srd + 1) % 4
                    if self.balance_cars
                    else np.random.choice(range(4))
                )
                erd = (
                    srd_new + np.random.choice([1, 2, 3])
                    if (
                        (hasattr(self, "turns") and self.turns)
                    )
                    else (srd_new + 2)
                )
                erd = erd % 4

                spos, epos, orient, dorient = self._sample_vehicle_on_road(
                    srd_new, erd
                )
                if vehicle is None:
                    vehicle = BatchedVehicle(
                        position=spos,
                        orientation=orient,
                        destination=epos,
                        dest_orientation=dorient,
                        dimensions=dims,
                        initial_speed=torch.zeros(1, 1),
                        name="agent",
                        vision_range=self.vision_range,
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
            srd = srd_new
            self.srd.append(srd)
            self.erd.append(erd)

        vehicle.add_bool_buffer(self.bool_buffer)

        self.world.add_vehicle(vehicle)
        self.store_dynamics(vehicle)
        self.agents[vehicle.name] = vehicle

        self.original_distances = vehicle.distance_from_destination()

    def store_dynamics(self, vehicle):
        self.dynamics = BicycleKinematicsModel(
            dim=vehicle.dimensions[:, 0], v_lim=torch.ones(self.nagents) * 8.0
        )

    def reset(self):
        # Keep the environment fixed for now
        world, config = self.generate_world_without_agents()
        for k, v in config.items():
            setattr(self, k, v)
        self.world = world
        self.add_vehicles_to_world()

        self.queue1 = deque(maxlen=self.history_len)
        self.queue2 = deque(maxlen=self.history_len)

        return super().reset()


class MultiAgentRoadIntersectionBicycleKinematicsDiscreteEnvironment(
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment
):
    def configure_action_space(self):
        self.max_accln = 1.5
        self.max_steering = 0.1
        actions = list(
            product(
                torch.arange(
                    -self.max_steering, self.max_steering + 0.01, 0.05
                ),
                torch.arange(-self.max_accln, self.max_accln + 0.05, 0.5),
            )
        )
        self.action_list = torch.as_tensor(actions)

    def get_action_space(self):
        self.normalization_factor = torch.as_tensor(
            [self.max_steering, self.max_accln]
        )
        return Discrete(self.action_list.size(0))

    def discrete_to_continuous_actions(self, action: torch.Tensor):
        return self.action_list[action]
