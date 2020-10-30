import math
import random
from itertools import product

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple

from sdriving.environments.intersection import (
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment,
)
from sdriving.tsim import (
    BatchedVehicle,
    FixedTrackAccelerationModel,
    World,
    angle_normalize,
    generate_intersection_world_4signals,
    generate_intersection_world_12signals,
    intervehicle_collision_check,
)


class MultiAgentRoadIntersectionFixedTrackEnvironment(
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment
):
    def __init__(
        self,
        *args,
        turns: bool = False,
        default_color: bool = True,
        learn_right_of_way: bool = False,
        **kwargs
    ):
        self.lane_side = 1
        self.turns = turns
        self.default_color = default_color
        self.learn_right_of_way = learn_right_of_way
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
                merge_same_signals=self.learn_right_of_way,
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


class MultiAgentRoadIntersectionFixedTrackDiscreteCommunicationEnvironment(
    MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment
):
    def configure_action_space(self):
        self.max_accln = 1.5
        self.action_list = torch.arange(
            -self.max_accln, self.max_accln + 0.05, step=0.25
        ).unsqueeze(1)
        accln_values = (
            torch.arange(-self.max_accln, self.max_accln + 0.05, step=0.25)
            .numpy()
            .tolist()
        )
        comm_values = [0.0, 1.0]
        self.action_list = torch.as_tensor(
            list(product(accln_values, comm_values))
        ).float()
        if not self.turns:
            self.action_list = torch.cat(
                [
                    torch.zeros((self.action_list.size(0), 1)).type_as(
                        self.action_list
                    ),
                    self.action_list,
                ],
                dim=-1,
            )

    def discrete_to_continuous_actions(self, action: torch.Tensor):
        action = self.action_list[action]
        comm = action[:, 1:] if self.turns else action[:, 2:]
        pos = self.agents["agent"].position
        self.world.broadcast_data(comm, pos)
        return (
            action[:, :1] if self.turns else action[:, :2]
        )  # Only return controls

    def generate_world_without_agents(self):
        world, params = super().generate_world_without_agents()
        if hasattr(self, "actual_nagents"):
            # this is needed for the first call to this function
            world.initialize_communication_channel(self.actual_nagents, 1)
        return world, params

    def get_observation_space(self):
        return Tuple(
            [
                Box(
                    low=np.array(
                        [0.0, -1.0, -math.pi, 0.0, 0.0] * self.history_len
                    ),
                    high=np.array(
                        [1.0, 1.0, math.pi, np.inf, 1.0] * self.history_len
                    ),
                ),
                Box(0.0, np.inf, shape=(self.npoints * self.history_len,)),
            ]
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

        comm_data = self.world.get_broadcast_data_all_agents()

        obs = torch.cat(
            [ts, speed / self.dynamics.v_lim, head, inv_dist, comm_data], -1
        )
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
