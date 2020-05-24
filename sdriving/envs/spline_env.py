import itertools
import math
import random
from collections import deque

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple

from sdriving.envs.base_hierarchical import BaseHierarchicalEnv
from sdriving.envs.intersection_env import RoadIntersectionControlEnv
from sdriving.trafficsim.common_networks import (
    generate_intersection_world_4signals,
    generate_intersection_world_12signals,
)
from sdriving.trafficsim.dynamics import (
    SplineAccelerationModel as VehicleDynamics,
)
from sdriving.trafficsim.utils import angle_normalize
from sdriving.trafficsim.vehicle import Vehicle
from sdriving.trafficsim.world import World


class SplineRoadIntersectionAccelerationControlEnv(
    BaseHierarchicalEnv, RoadIntersectionControlEnv,
):
    def __init__(
        self,
        npoints: int = 100,
        horizon: int = 200,
        tolerance: float = 4.0,
        object_collision_penalty: float = 1.0,
        agents_collision_penalty: float = 1.0,
        goal_reach_bonus: float = 1.0,
        history_len: int = 5,
        time_green: int = 100,
        nagents: int = 4,
        device=torch.device("cpu"),
        lidar_range: float = 50.0,
        lidar_noise: float = 0.0,
        mode: int = 1,
        balance_cars: bool = True,
        road_params: dict = dict(),
        cp_num: int = 3,
        p_num: int = 500,
        meta_controller_discretization: float = 0.2,
        acceleration_discretization: float = 0.5,
        steering_discretization: float = 0.05,
        max_acceleration: float = 1.5,
        min_acceleration: float = -1.5,
        max_steering: float = -0.1,
        min_steering: float = 0.1,
    ):
        self.npoints = npoints
        self.goal_reach_bonus = goal_reach_bonus
        self.history_len = history_len
        self.time_green = time_green
        self.device = device

        # Action Parameters
        self.acceleration_discretization = acceleration_discretization
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.steeting_discretization = steering_discretization
        self.max_steering = max_steering
        self.min_steering = min_steering
        self.meta_controller_discretization = meta_controller_discretization

        # Road Parameters
        self.mean_length = road_params.get("mean_length", 40.0)
        self.std_length = road_params.get("std_lenth", 30.0)
        self.mean_width = road_params.get("mean_width", 15.0)
        self.std_width = road_params.get("std_width", 15.0)

        world = self.generate_world_without_agents()
        BaseHierarchicalEnv.__init__(
            self,
            world,
            nagents,
            horizon,
            tolerance,
            object_collision_penalty,
            agents_collision_penalty,
            cp_num,
            p_num,
        )

        self.queue1 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.queue2 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.lidar_range = lidar_range
        self.lidar_noise = lidar_noise
        self.mode = mode
        self.prev_actions = {
            a_id: torch.zeros(2) for a_id in self.get_agent_ids_list()
        }
        self.curr_actions = {a_id: None for a_id in self.get_agent_ids_list()}
        self.balance_cars = balance_cars

    def generate_world_without_agents(self):
        # FIXME: This is not the std and mean :P
        self.length = (
            torch.rand(1) * self.std_length + self.mean_length
        ).item()
        self.width = (torch.rand(1) * self.std_width + self.mean_width).item()
        return generate_intersection_world_4signals(
            length=self.length,
            road_width=self.width,
            name="traffic_signal_world",
            time_green=self.time_green,
            ordering=random.choice([0, 1]),
        )

    def _configure_metacontroller_actions(self):
        x_vals = np.arange(-1.0, 1.01, self.meta_controller_discretization)
        y_vals = np.arange(-1.0, 1.01, self.meta_controller_discretization)
        xy_vals = [
            torch.as_tensor(xy).unsqueeze(0)
            for xy in itertools.product(x_vals, y_vals)
        ]
        self.meta_controller_actions = list(
            itertools.product(
                *[xy_vals for _ in range(self.cp_num - 1)]
            )
        )

    def _configure_controller_actions(self):
        accln_vals = np.arange(
            self.min_acceleration,
            self.max_acceleration,
            self.acceleration_discretization,
        )
        self.controller_actions = [
            torch.as_tensor([ac]).unsqueeze(0) for ac in accln_vals
        ]
        self.actions_list = self.controller_actions

    def _get_meta_controller_action_space(self):
        return Discrete(len(self.meta_controller_actions))

    def _get_controller_action_space(self):
        return Discrete(len(self.controller_actions))

    def _get_meta_controller_observation_space(self):
        min_width = self.mean_width
        max_width = self.mean_width + self.std_width
        min_length = self.mean_length
        max_length = self.mean_length + self.std_length
        # {x_s, y_s, x_g, y_g, width, length}
        return Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, min_width, min_length]),
            high=np.array([1.0, 1.0, 1.0, 1.0, max_width, max_length]),
        )

    def _get_controller_observation_space(self):
        return Tuple(
            [
                Box(
                    # Signal, Velocity, Heading, Inverse Distance
                    low=np.array([0.0, -1.0, -1.0, 0.0] * self.history_len),
                    high=np.array([1.0, 1.0, 1.0, np.inf] * self.history_len),
                ),
                Box(0.0, np.inf, shape=(self.npoints * self.history_len,)),
            ]
        )

    def get_state_single_agent_metacontroller(self, a_id: str):
        x_max = y_max = self.width / 2 + self.length
        agent = self.agents[a_id]["vehicle"]
        return torch.as_tensor(
            [
                agent.position[0] / x_max,
                agent.position[1] / y_max,
                agent.destination[0] / x_max,
                agent.destination[1] / y_max,
                self.width,
                self.length,
            ]
        )

    def get_state_single_agent(self, a_id: str):
        agent = self.agents[a_id]["vehicle"]
        v_lim = self.agents[a_id]["v_lim"]
        dynamics = self.agents[a_id]["dynamics"]

        dest = dynamics.get_next_target()
        inv_dist = 1 / agent.distance_from_point(dest)

        pt1, pt2 = self.get_next_two_goals(a_id)
        obs = [
            self.world.get_traffic_signal(
                pt1, pt2, agent.position, agent.vision_range
            ),
            agent.speed / v_lim,
            agent.optimal_heading_to_point(dest) / math.pi,
            inv_dist if torch.isfinite(inv_dist) else 0.0,
        ]
        cur_state = [
            torch.as_tensor(obs),
            1 / self.world.get_lidar_data(agent.name, self.npoints),
        ]

        # TODO: Modify to use conditional gaussian noise
        if self.lidar_noise != 0.0:
            cur_state[1] *= torch.rand(self.npoints) > self.lidar_noise

        while len(self.queue1[a_id]) <= self.history_len - 1:
            self.queue1[a_id].append(cur_state[0])
            self.queue2[a_id].append(cur_state[1])
        self.queue1[a_id].append(cur_state[0])
        self.queue2[a_id].append(cur_state[1])

        return (
            torch.cat(list(self.queue1[a_id])),
            torch.cat(list(self.queue2[a_id])),
        )

    def add_vehicle_path(self, *args, **kwargs):
        return RoadIntersectionControlEnv.add_vehicle_path(
            self, *args, dynamics_model=VehicleDynamics, **kwargs
        )

    def reset(self):
        self.world = self.generate_world_without_agents()
        self.world.compile()

        self.queue1 = {
            a_id: deque(maxlen=self.history_len)
            for a_id in self.get_agent_ids_list()
        }
        self.queue2 = {
            a_id: deque(maxlen=self.history_len)
            for a_id in self.get_agent_ids_list()
        }

        if self.nagents == 1:
            self.setup_nagents_1()
        elif self.nagents == 2:
            self.setup_nagents_2()
        else:
            self.setup_nagents(self.nagents)

        return BaseHierarchicalEnv.reset(self)
