import math
from collections import deque
from itertools import product

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple

from sdriving.environments.base_env import BaseMultiAgentDrivingEnvironment
from sdriving.tsim import (
    BatchedVehicle,
    BicycleKinematicsModel,
    Pedestrians,
    Road,
    RoadNetwork,
    SplineModel,
    World,
    angle_normalize,
    generate_intersection_world_4signals,
    intervehicle_collision_check,
)


class MultiAgentHighwayBicycleKinematicsModel(
    BaseMultiAgentDrivingEnvironment
):
    def __init__(
        self,
        npoints: int = 360,
        horizon: int = 200,
        timesteps: int = 25,
        history_len: int = 5,
        nagents: int = 1,
        device: torch.device = torch.device("cpu"),
        lidar_noise: float = 0.0,
        lateral_noise_variance: float = 0.8,
    ):
        self.npoints = npoints
        self.history_len = history_len
        self.device = device
        world, config = self.generate_world_without_agents()
        for k, v in config.items():
            setattr(self, k, v)
        super().__init__(world, nagents, horizon, timesteps, device)
        self.queue1 = None
        self.queue2 = None
        self.lidar_noise = lidar_noise

        bool_buffer = torch.ones(self.nagents * 4, self.nagents * 4)
        for i in range(0, self.nagents * 4, 4):
            bool_buffer[i : (i + 4), i : (i + 4)] -= 1
        self.bool_buffer = bool_buffer.bool()

        self.lateral_noise_variance = lateral_noise_variance

    def generate_world_without_agents(self):
        network = RoadNetwork()
        length = 250.0
        width = 25.0
        network.add_road(
            Road(
                f"highway",
                torch.zeros(1, 2),
                length,
                width,
                torch.zeros(1, 1),
                can_cross=[False] * 4,
                has_endpoints=[True, False, True, False],
            )
        )
        return (
            World(
                network,
                xlims=(-length / 2 - 10, length / 2 + 10),
                ylims=(-length / 2 - 10, length / 2 + 10),
            ),
            {"length": length, "width": width},
        )

    def get_observation_space(self):
        return Tuple(
            [
                Box(
                    low=np.array([0.0, -1.0, 0.5, 0.5] * self.history_len),
                    high=np.array([1.0, 1.0, 1.0, 1.0] * self.history_len),
                ),
                Box(0.0, np.inf, shape=(self.npoints * self.history_len,)),
            ]
        )

    def get_action_space(self):
        return Box(low=np.array([-0.1, -1.0]), high=np.array([0.1, 1.0]))

    def get_state(self):
        a_ids = self.get_agent_ids_list()

        dist = torch.cat(
            [
                (v.destination[:, 0:1] - v.position[:, 0:1]).abs()
                for v in self.agents.values()
            ]
        )
        inv_dist = 1 / dist.clamp(min=1.0)
        speed = torch.cat([self.agents[v].speed for v in a_ids])

        obs = torch.cat(
            [
                inv_dist,
                speed / self.dynamics.v_lim,
                self.accln_rating,
                self.vel_rating,
            ],
            -1,
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
            return (obs, lidar), self.agent_names

    def vehicle_collision_check(self, vehicle):
        # if self.lateral_noise_variance == 0.0:
        #     return vehicle.collision_check()
        # Hopefully this doesn't get messed up by some inplace operation
        # position_dup = vehicle.position.clone()
        # position_noise = (
        #     torch.rand_like(position_dup[:, 1]) * self.lateral_noise_variance
        # )
        # vehicle.position[:, 1] += position_noise
        # vehicle.cached_coordinates = False
        collision = vehicle.collision_check()
        # vehicle.position = position_dup
        # vehicle.cached_coordinates = False
        return collision

    def get_reward(self, new_collisions: torch.Tensor, action: torch.Tensor):
        a_ids = self.get_agent_ids_list()

        # Distance from destination
        distances = torch.cat(
            [
                v.destination[:, 0:1] - v.position[:, 0:1]
                for v in self.agents.values()
            ]
        )

        # Agent Speeds
        speeds = torch.cat([self.agents[v].speed for v in a_ids])

        # Goal Reach Bonus
        reached_goal = distances <= 0.0
        distances = distances.abs()
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
        penalty = (
            new_collisions.float()
            + new_collisions
            * distances
            * (self.horizon - self.nsteps - 1)
            / self.horizon
        )
        self.collision_vector += new_collisions

        return (
            -distances * ~self.collision_vector / self.horizon
            - (speeds / 8.0).abs() * self.completion_vector / self.horizon
            - penalty
            + goal_reach_bonus
        )

    def add_vehicles_to_world(self):
        dims = torch.as_tensor([[4.48, 2.2]]).repeat(self.nagents, 1)

        self.max_accln = 3.0
        self.max_velocity = 16.0

        diffs = torch.cumsum(
            torch.as_tensor([0.0] + [10.0] * (self.nagents - 1)).unsqueeze(1),
            dim=0,
        )
        diffs = torch.cat([diffs, torch.zeros(self.nagents, 1)], dim=-1)
        spos = torch.as_tensor([[-self.length / 2 + 30.0, 0.0]]) + diffs

        epos = torch.as_tensor([[self.length / 2 - 50.0, 0.0]]).repeat(
            self.nagents, 1
        )

        orient = torch.zeros(self.nagents, 1)
        dorient = torch.zeros(self.nagents, 1)

        vehicle = BatchedVehicle(
            position=spos,
            orientation=orient,
            destination=epos,
            dest_orientation=dorient,
            dimensions=dims,
            initial_speed=torch.zeros(self.nagents, 1),
            name="agent",
        )

        vehicle.add_bool_buffer(self.bool_buffer)

        self.accln_rating = (torch.rand(self.nagents, 1) + 1) * 0.5
        self.vel_rating = self.accln_rating

        # Just for coloring the vehicles
        self.world.initialize_communication_channel(self.actual_nagents, 1)
        self.world.broadcast_data(self.accln_rating, vehicle.position)

        self.world.add_vehicle(vehicle, False)
        self.store_dynamics(vehicle)
        self.agents[vehicle.name] = vehicle

        self.original_distances = vehicle.distance_from_destination()

    def store_dynamics(self, vehicle):
        self.dynamics = BicycleKinematicsModel(
            dim=vehicle.dimensions[:, 0],
            v_lim=self.vel_rating[:, 0] * self.max_velocity,
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

    def discrete_to_continuous_actions(self, actions: torch.Tensor):
        self.world.broadcast_data(
            self.accln_rating, self.agents["agent"].position
        )
        actions[:, 1:] = (
            actions[:, 1:]
            * self.max_accln
            * self.accln_rating.to(actions.device)
        )
        return actions


class MultiAgentHighwayBicycleKinematicsDiscreteModel(
    MultiAgentHighwayBicycleKinematicsModel
):
    def configure_action_space(self):
        self.max_accln = 3.0
        self.max_steering = 0.1
        actions = list(
            product(
                torch.arange(
                    -self.max_steering, self.max_steering + 0.01, 0.05
                ),
                torch.arange(-1, 1 + 0.05, 0.25),
            )
        )
        self.action_list = torch.as_tensor(actions)

    def get_action_space(self):
        self.normalization_factor = torch.as_tensor(
            [self.max_steering, self.max_accln]
        )
        return Discrete(self.action_list.size(0))

    def discrete_to_continuous_actions(self, actions: torch.Tensor):
        self.world.broadcast_data(
            self.accln_rating, self.agents["agent"].position
        )
        actions = self.action_list[actions]
        actions[:, 1:] = (
            actions[:, 1:]
            * self.max_accln
            * self.accln_rating.to(actions.device)
        )
        return actions


class MultiAgentHighwayPedestriansFixedTrackDiscreteModel(
    MultiAgentHighwayBicycleKinematicsDiscreteModel
):
    def configure_action_space(self):
        self.max_accln = 3.0
        self.max_steering = 0.1
        actions = torch.arange(
            -self.max_accln, self.max_accln + 0.05, 0.25
        ).unsqueeze(1)
        self.action_list = torch.cat(
            [torch.zeros(actions.size(0), 1), actions], dim=-1
        )

    def discrete_to_continuous_actions(self, actions: torch.Tensor):
        self.world.broadcast_data(
            self.accln_rating, self.agents["agent"].position
        )
        return self.action_list[actions]

    def generate_world_without_agents(self):
        network = RoadNetwork()
        length = 140.0
        width = 25.0
        network.add_road(
            Road(
                f"highway",
                torch.zeros(1, 2),
                length,
                width,
                torch.zeros(1, 1),
                can_cross=[False] * 4,
                has_endpoints=[True, False, True, False],
            )
        )
        return (
            World(
                network,
                xlims=(-length / 2 - 10, length / 2 + 10),
                ylims=(-length / 2 - 10, length / 2 + 10),
            ),
            {"length": length, "width": width},
        )

    def add_vehicles_to_world(self):
        self.max_velocity = 10.0

        vehicle = None
        dims = torch.as_tensor([[4.48, 2.2]])
        d1 = torch.as_tensor([[-self.length * 0.45, self.width * 0.375]])
        d2 = torch.as_tensor([[-self.length * 0.3, -self.width * 0.375]])
        epos = torch.as_tensor([[self.length * 0.3, 0.0]])
        orient = dorient = torch.zeros(1, 1)
        for _ in range(self.actual_nagents):
            successful_placement = False
            while not successful_placement:
                spos = torch.rand(1, 2) * (d1 - d2) + d2
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

        vehicle.add_bool_buffer(self.bool_buffer)

        self.accln_rating = (torch.rand(self.nagents, 1) + 1) * 0.5
        self.vel_rating = torch.ones_like(self.accln_rating)

        # Just for coloring the vehicles
        self.world.initialize_communication_channel(self.actual_nagents, 1)
        self.world.broadcast_data(self.accln_rating, vehicle.position)

        self.world.add_vehicle(vehicle, False)
        self.store_dynamics(vehicle)
        self.agents[vehicle.name] = vehicle

        self.original_distances = vehicle.distance_from_destination()

        _pos = []
        # dims = torch.ones(10, 2) * 0.8
        # orientation = torch.ones(10, 1) * math.pi / 2
        for i in range(10):
            if torch.rand(1) < 0.1:
                continue
            pos = torch.zeros(1, 2)
            # CrossWalk is from -10.0 to 10.0
            pos[0, 0] = (torch.rand(1) * 2.0 - 1.0) * 10.0
            pos[0, 1] = -torch.rand(1) * self.width / 2
            _pos.append(pos)
        pedestrians = Pedestrians(
            torch.cat(_pos),
            torch.ones(len(_pos), 1) * 0.8,
            torch.ones(len(_pos), 1) * math.pi / 2,
            torch.rand(len(_pos), 1) + 1.0,
        )
        self.world.add_object(pedestrians)


class MultiAgentHighwaySplineAccelerationDiscreteModel(
    MultiAgentHighwayBicycleKinematicsModel
):
    def configure_action_space(self):
        self.max_accln = 3.0
        self.action_list = torch.arange(
            -self.max_accln, self.max_accln + 0.05, step=0.25
        ).unsqueeze(1)

    def get_observation_space(self):
        return (
            Box(low=np.array([0.5]), high=np.array([1.0])),
            Tuple(
                [
                    Box(
                        low=np.array([0.0, -1.0] * self.history_len),
                        high=np.array([1.0, 1.0] * self.history_len),
                    ),
                    Box(0.0, np.inf, shape=(self.npoints * self.history_len,)),
                ]
            ),
        )

    def get_action_space(self):
        return (
            Box(low=np.array([-0.75]), high=np.array([0.75])),
            Discrete(self.action_list.size(0)),
        )

    def discrete_to_continuous_actions(self, action: torch.Tensor):
        self.world.broadcast_data(
            self.accln_rating, self.agents["agent"].position
        )
        action = self.action_list[action]
        return action * self.max_accln * self.accln_rating.to(action.device)

    def discrete_to_continuous_actions_v2(self, action: torch.Tensor):
        return action

    def _get_spline_state(self):
        self.got_spline_state = True
        return self.accln_rating, self.agent_names

    def get_state(self):
        if not self.got_spline_state:
            return self._get_spline_state()

        a_ids = self.get_agent_ids_list()

        dist = torch.cat(
            [
                (v.destination[:, 0:1] - v.position[:, 0:1]).abs()
                for v in self.agents.values()
            ]
        )
        inv_dist = 1 / dist.clamp(min=1.0)
        speed = torch.cat([self.agents[v].speed for v in a_ids])

        obs = torch.cat([inv_dist, speed / self.dynamics.v_lim], -1)
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
            return (obs, lidar), self.agent_names

    @torch.no_grad()
    def step(
        self,
        stage: int,  # Possible Values [0, 1]
        action: torch.Tensor,
        render: bool = False,
        **render_kwargs,
    ):
        assert stage in [0, 1]

        if stage == 1:
            return super().step(action, render, **render_kwargs)

        action = self.discrete_to_continuous_actions_v2(action)
        action = action.to(self.world.device)

        vehicle = self.agents["agent"]
        pos = vehicle.position
        farthest_pt = torch.cat(
            [torch.full((pos.size(0), 1), -self.length / 2), pos[:, 1:]],
            dim=-1,
        ).to(pos.device)
        mid_point_x = pos[:, :1] + 50.0
        mid_point_y = action * self.width / 2
        mid_point = torch.cat([mid_point_x, mid_point_y], dim=-1)
        last_pt = torch.cat(
            [torch.full((pos.size(0), 1), self.length / 2), mid_point_y],
            dim=-1,
        )

        action = torch.cat(
            [x.unsqueeze(1) for x in [pos, mid_point, last_pt, farthest_pt]],
            dim=1,
        )

        if self.lateral_noise_variance != 0.0:
            noise = (
                torch.randn(action.shape[0], 2, 2)
                * self.lateral_noise_variance
            )
            noise.clamp_(-5.0, 5.0)
            action[:, 1:3, :] += noise

        self.dynamics = SplineModel(
            action, v_lim=self.vel_rating[:, 0] * self.max_velocity
        )

        return self.get_state()

    def reset(self):
        self.got_spline_state = False
        return super().reset()


class MultiAgentHighwayPedestriansSplineAccelerationDiscreteModel(
    MultiAgentHighwaySplineAccelerationDiscreteModel
):
    def generate_world_without_agents(self):
        network = RoadNetwork()
        length = 140.0
        width = 25.0
        network.add_road(
            Road(
                f"highway",
                torch.zeros(1, 2),
                length,
                width,
                torch.zeros(1, 1),
                can_cross=[False] * 4,
                has_endpoints=[True, False, True, False],
            )
        )
        return (
            World(
                network,
                xlims=(-length / 2 - 10, length / 2 + 10),
                ylims=(-length / 2 - 10, length / 2 + 10),
            ),
            {"length": length, "width": width},
        )

    def add_vehicles_to_world(self):
        dims = torch.as_tensor([[4.48, 2.2]]).repeat(self.nagents, 1)

        self.max_accln = 3.0
        self.max_velocity = 16.0

        diffs = torch.cumsum(
            torch.as_tensor([0.0] + [10.0] * (self.nagents - 1)).unsqueeze(1),
            dim=0,
        )
        diffs = torch.cat([diffs, torch.zeros(self.nagents, 1)], dim=-1)
        spos = torch.as_tensor([[-self.length / 2 + 30.0, 0.0]]) + diffs

        epos = torch.as_tensor([[self.length / 2 - 50.0, 0.0]]).repeat(
            self.nagents, 1
        )

        orient = torch.zeros(self.nagents, 1)
        dorient = torch.zeros(self.nagents, 1)

        vehicle = BatchedVehicle(
            position=spos,
            orientation=orient,
            destination=epos,
            dest_orientation=dorient,
            dimensions=dims,
            initial_speed=torch.zeros(self.nagents, 1),
            name="agent",
        )

        vehicle.add_bool_buffer(self.bool_buffer)

        self.accln_rating = (torch.rand(self.nagents, 1) + 1) * 0.5
        self.vel_rating = self.accln_rating

        # Just for coloring the vehicles
        self.world.initialize_communication_channel(self.actual_nagents, 1)
        self.world.broadcast_data(self.accln_rating, vehicle.position)

        self.world.add_vehicle(vehicle, False)
        self.store_dynamics(vehicle)
        self.agents[vehicle.name] = vehicle

        self.original_distances = vehicle.distance_from_destination()

        _pos = []
        # dims = torch.ones(10, 2) * 0.8
        # orientation = torch.ones(10, 1) * math.pi / 2
        for i in range(10):
            if torch.rand(1) < 0.1:
                continue
            pos = torch.zeros(1, 2)
            # CrossWalk is from -10.0 to 10.0
            pos[0, 0] = (torch.rand(1) * 2.0 - 1.0) * 10.0
            pos[0, 1] = -torch.rand(1) * self.width / 2
            _pos.append(pos)
        pedestrians = Pedestrians(
            torch.cat(_pos),
            torch.ones(len(_pos), 1) * 0.8,
            torch.ones(len(_pos), 1) * math.pi / 2,
            torch.rand(len(_pos), 1) + 1.0,
        )
        self.world.add_object(pedestrians)
