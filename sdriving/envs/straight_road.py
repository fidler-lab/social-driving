import itertools
import math
import random
from collections import deque

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple
from sdriving.envs.base_env import BaseEnv
from sdriving.trafficsim import (
    generate_straight_road,
    BicycleKinematicsModel as VehicleDynamics,
    angle_normalize,
    Vehicle,
    World,
    Pedestrian
)


class StraightRoadPedestrianAvoidanceEnv(BaseEnv):
    def __init__(
        self,
        nagents: int = 1,
        npoints: int = 360,
        horizon: int = 200,
        tolerance: float = 0.0,
        object_collision_penalty: float = 1.0,
        agents_collision_penalty: float = 1.0,
        goal_reach_bonus: float = 1.0,
        history_len: int = 5,
        time_green: int = 75,
        device=torch.device("cpu"),
        lidar_range: float = 50.0,
        lidar_noise: float = 0.0,
    ):
        self.npoints = npoints
        self.goal_reach_bonus = goal_reach_bonus
        self.history_len = history_len
        self.time_green = time_green
        self.device = device
        world = self.generate_world_without_agents()
        super().__init__(
            world,
            nagents,
            horizon,
            tolerance,
            object_collision_penalty,
            agents_collision_penalty,
            False,
        )
        # We need to store the history to stack them. 1 queue stores the
        # observation and the other stores the lidar data
        self.queue1 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.queue2 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.lidar_range = lidar_range
        self.lidar_noise = lidar_noise
        self.prev_actions = {
            a_id: torch.zeros(2) for a_id in self.get_agent_ids_list()
        }
        self.curr_actions = {a_id: None for a_id in self.get_agent_ids_list()}

    def generate_world_without_agents(self):
        self.length = 140.0
        self.width = 20.0
        net = generate_straight_road(
            length=self.length,
            road_width=self.width,
            name="pedestrian_world",
        )
        net.construct_graph()
        return World(net)

    def configure_action_list(self):
        self.actions_list = [
            torch.as_tensor(ac).unsqueeze(0)
            for ac in itertools.product(
                [-0.1, -0.05, 0.0, 0.05, 0.1], [-1.5, -0.75, 0.0, 0.75, 1.5],
            )
        ]

    def get_action_space(self):
        self.max_accln = 1.5
        self.max_steering = 0.1
        return Discrete(len(self.actions_list))
    
    def get_observation_space(self):
        return Tuple(
            [
                Box(
                    # Velocity Diff, Optimal Heading, Inverse Distance
                    low=np.array([-1.0, -1.0, 0.0] * self.history_len),
                    high=np.array([1.0, 1.0, np.inf] * self.history_len),
                ),
                Box(0.0, np.inf, shape=(self.npoints * self.history_len,)),
            ]
        )
    
    def add_vehicle(
        self,
        a_id,
        pos,
        v_lim,
        orientation,
        dest,
        dest_orientation,
        dynamics_model=VehicleDynamics,
        dynamics_kwargs={},
    ):
        rname = "pedestrian_world"
        vehicle = Vehicle(
            pos,
            orientation,
            destination=dest,
            dest_orientation=dest_orientation,
            name=a_id,
            max_lidar_range=self.lidar_range,
            initial_speed=torch.rand(1).item() * v_lim
        )
        dynamics = dynamics_model(
            dim=[vehicle.dimensions[0]],
            v_lim=[v_lim.item()],
            **dynamics_kwargs,
        )

        # Register the vehicle with the world
        self.world.add_vehicle(vehicle, rname, v_lim=v_lim.item())

        self.agents[a_id] = {
            "vehicle": vehicle,
            "done": False,
            "v_lim": v_lim.item(),
            "dynamics": dynamics,
            "road name": rname,
            "original_destination": dest,
            "dest_orientation": dest_orientation,
            "original_distance": vehicle.distance_from_destination(),
            "goal_reach_bonus": False,
        }
        
    def get_state_single_agent(self, a_id):
        agent = self.agents[a_id]["vehicle"]
        v_lim = self.agents[a_id]["v_lim"]

        dest = agent.destination

        inv_dist = 1 / agent.distance_from_point(dest)
        obs = [
            (agent.speed / v_lim).unsqueeze(0),
            agent.optimal_heading_to_point(dest).unsqueeze(0) / math.pi,
            inv_dist.unsqueeze(0)
            if torch.isfinite(inv_dist)
            else torch.zeros(1),
        ]
        obs = torch.cat(obs)
        cur_state = [
            obs,
            1 / self.world.get_lidar_data(agent.name, self.npoints),
        ]

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

    def handle_goal_tolerance(self, agent):
        if (
            agent["vehicle"].distance_from_destination()
            < 4.0  # self.goal_tolerance
        ) or (
            self.world.road_network.is_perpendicular(
                self.world.vehicles[agent["vehicle"].name].road,
                agent["vehicle"].destination,
                agent["vehicle"].position,
            )
        ):
            agent["vehicle"].destination = agent["vehicle"].position
            agent["vehicle"].dest_orientation = agent["vehicle"].orientation
            if not agent["goal_reach_bonus"]:
                agent["goal_reach_bonus"] = True
                return self.goal_reach_bonus
            else:
                return -torch.abs(
                    (
                        agent["vehicle"].speed
                        / (agent["dynamics"].v_lim * self.horizon)
                    )
                ).item()
        return 0.0
    
    def distance_reward_function(self, agent):
        return torch.abs(
            agent["vehicle"].destination[0] - agent["vehicle"].position[0]
        ) / (self.length * self.horizon)
    
    def reset(self):
        self.world = self.generate_world_without_agents()

        self.queue1 = {
            a_id: deque(maxlen=self.history_len)
            for a_id in self.get_agent_ids_list()
        }
        self.queue2 = {
            a_id: deque(maxlen=self.history_len)
            for a_id in self.get_agent_ids_list()
        }

        self.add_vehicle(
            "agent_0",
            torch.as_tensor([
                -self.length * 0.5 * 0.75 + torch.randn(1) * 3.0,
                (torch.rand(1).item() - 0.5) * self.width * 0.75
            ]),
            torch.as_tensor(12.0),
            0.0,
            torch.as_tensor([self.length * 0.5 * 0.75, 0.0]),
            0.0,
        )
        
        for i in range(10):
            if torch.rand(1) < 0.1:
                continue
            pos = torch.zeros(2)
            # CrossWalk is from -10.0 to 10.0
            pos[0] = (torch.rand(1) * 2.0 - 1.0) * 10.0
            pos[1] = -torch.rand(1) * self.width / 2
            pedestrian = Pedestrian(
                f"person_{i}",
                pos,
                orientation=torch.as_tensor(math.pi / 2),
                velocity=torch.rand(1) + 1.0
            )
            self.world.add_object(pedestrian)

        self.world.compile()
        return super().reset()
    
    def transform_state_action(self, actions, states, timesteps):
        action = []
        start_state = []
        for a_id in self.get_agent_ids_list():
            self.check_in_space(self.action_space, actions[a_id])
            self.check_in_space(self.observation_space, states[a_id])
            agent = self.agents[a_id]["vehicle"]
            action.append(self.actions_list[actions[a_id]])
            start_state.append(
                torch.cat(
                    [
                        agent.position,
                        agent.speed.unsqueeze(0),
                        agent.orientation.unsqueeze(0),
                    ]
                ).unsqueeze(0)
            )
        action = torch.cat(action, dim=0)
        state = torch.cat(start_state, dim=0)

        un_action = action.unsqueeze(1)
        nominal_actions = [un_action for _ in range(timesteps)]
        nominal_states = [state.unsqueeze(1)]

        for _ in range(timesteps):
            state = self.world.global_dynamics(state, action)
            nominal_states.append(state.unsqueeze(1))

        nominal_actions = torch.cat(nominal_actions, dim=1)
        nominal_states = torch.cat(nominal_states, dim=1)

        return_val = {}
        for i, a_id in enumerate(self.get_agent_ids_list()):
            return_val[a_id] = (
                nominal_states[i, :, :],
                nominal_actions[i, :, :],
            )
            self.curr_actions[a_id] = action[i]
        return None, None, return_val

    def render(self, *args, **kwargs):
        render_lidar = kwargs.get("render_lidar", True)
        kwargs.update({"render_lidar": render_lidar})
        self.world.render(*args, **kwargs)
