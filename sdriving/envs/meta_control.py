import itertools
import math
import random

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple
from sdriving.envs.base_env import BaseEnv
from sdriving.trafficsim.common_networks import generate_straight_road
from sdriving.trafficsim.controller import MPCControllerWrapper
from sdriving.trafficsim.dynamics import (
    BicycleKinematicsModel as VehicleDynamics,
)
from sdriving.trafficsim.vehicle import Vehicle
from sdriving.trafficsim.world import World


class MetaControlEnv(BaseEnv):
    def __init__(
        self,
        npoints: int = 100,
        horizon: int = 400,
        tolerance: float = 0.0,
        object_collision_penalty: float = 1.0,
        agents_collision_penalty: float = 1.0,
    ):
        self.npoints = npoints
        world = self.generate_world_without_agents()
        super().__init__(
            world,
            1,
            horizon,
            tolerance,
            object_collision_penalty,
            agents_collision_penalty,
        )

    def generate_world_without_agents(self):
        length = torch.rand(1) * 100.0 + 100.0
        width = torch.rand(1) * 20.0 + 20.0
        if length > width:
            figsize = (10, 10 * width // length)
        else:
            figsize = (10 * length // width, 10)
        world = World(
            generate_straight_road(length=length, road_width=width),
            figsize=figsize,
        )
        world.road_network.construct_graph()
        return world

    def configure_action_list(self):
        # Action Space --> {Steering Direction, Acceleration}
        self.actions_list = [
            torch.as_tensor(ac)
            for ac in itertools.product(
                [-0.1, -0.05, 0.0, 0.05, 0.1], [-1.5, -0.75, 0.0, 0.75, 1.5],
            )
        ]

    def get_action_space(self):
        return Discrete(len(self.actions_list) + 1)

    def get_observation_space(self):
        return Tuple(
            [
                Box(
                    # Velocity Diff, Optimal Heading, Inverse Distance
                    low=np.array([-1.0, -1.0, 0.0]),
                    high=np.array([1.0, 1.0, np.inf]),
                ),
                Box(0.0, np.inf, shape=(self.npoints,)),
            ]
        )

    def add_vehicle(
        self, a_id, rname, pos, v_lim, orientation, dest, dest_orientation
    ):
        vehicle = Vehicle(
            pos,
            orientation,
            destination=dest,
            dest_orientation=dest_orientation,
            name=a_id,
        )
        dynamics = VehicleDynamics(
            dim=[vehicle.dimensions[0]], v_lim=[v_lim.item()]
        )
        controller = MPCControllerWrapper(self.world)

        # Register the vehicle with the world
        self.world.add_vehicle(vehicle, rname, v_lim=v_lim.item())

        # Store local references to the agent as well
        self.agents[a_id] = {
            "vehicle": vehicle,
            "done": False,
            "dynamics": dynamics,
            "controller": controller,
            "original_destination": dest,
            "dest_orientation": dest_orientation,
            "original_distance": vehicle.distance_from_destination(),
        }

    def get_state_single_agent(self, a_id):
        agent = self.agents[a_id]["vehicle"]
        pos = agent.position * 2
        dest = agent.destination * 2
        v_lim = self.agents[a_id]["dynamics"].v_lim
        inv_dist = 1 / agent.distance_from_destination()
        return (
            torch.as_tensor(
                [
                    (0.0 - agent.speed) / (2 * v_lim),
                    agent.optimal_heading() / math.pi,
                    inv_dist if torch.isfinite(inv_dist) else 0.0,
                ]
            ),
            1 / self.world.get_lidar_data(agent.name, self.npoints),
        )

    def transform_state_action_single_agent(
        self, a_id, action, state, timesteps
    ):
        agent = self.agents[a_id]["vehicle"]

        x, y = agent.position
        v = agent.speed
        t = agent.orientation

        if action > 0:
            action = self.actions_list[action - 1]
            start_state = torch.as_tensor([x, y, v, t])
            dynamics = self.agents[a_id]["dynamics"]
            nominal_states = [start_state.unsqueeze(0)]
            nominal_actions = [action.unsqueeze(0)]
            action = action.unsqueeze(0)
            for _ in range(timesteps):
                start_state = nominal_states[-1]
                new_state = dynamics(start_state, action)
                nominal_states.append(new_state.cpu())
                nominal_actions.append(action)
            nominal_states, nominal_actions = (
                torch.cat(nominal_states),
                torch.cat(nominal_actions),
            )
            na = torch.zeros(4).to(self.agents[a_id]["controller"].device)
            ns = torch.zeros(4).to(self.agents[a_id]["controller"].device)
            ex = (nominal_states, nominal_actions)
        else:
            na = torch.as_tensor(
                [*agent.destination, 0.0, agent.dest_orientation]
            )
            ns = torch.as_tensor([x, y, v, t])
            ex = None

        return na, ns, ex

    def handle_goal_tolerance(self, agent):
        if agent["vehicle"].distance_from_destination() < self.goal_tolerance:
            agent["vehicle"].destination = agent["vehicle"].position
            agent["vehicle"].dest_orientation = agent["vehicle"].orientation
        return 0.0

    def distance_reward_function(self, agent):
        return (
            torch.clamp(
                agent["vehicle"].distance_from_destination()
                / (agent["original_distance"] + 1e-8),
                0.0,
                2.0,
            )
            / self.horizon
        )

    def reset(self):
        # Randomize the environment as well
        self.world.reset()
        self.world = self.generate_world_without_agents()
        # Randomize start and end positions
        start, end = self.world.road_network.sample(2)
        # Register the vehicle
        self.add_vehicle(
            self.get_agent_ids_list()[0],
            start[0],
            start[1],
            torch.as_tensor(8.0),
            torch.rand(1) * 2.0 - 1.0,
            end[1],
            torch.rand(1) * 2.0 - 1.0,
        )

        return super().reset()
