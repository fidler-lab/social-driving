import itertools
import math
import random
from collections import deque

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple

from sdriving.envs.intersection_env import RoadIntersectionControlEnv
from sdriving.trafficsim.common_networks import (
    generate_intersection_world_4signals,
    generate_intersection_world_12signals,
)
from sdriving.trafficsim.controller import HybridController
from sdriving.trafficsim.dynamics import (
    BicycleKinematicsModel as VehicleDynamics,
)
from sdriving.trafficsim.utils import angle_normalize
from sdriving.trafficsim.vehicle import Vehicle
from sdriving.trafficsim.world import World


class RoadIntersectionContinuousFlowControlEnv(RoadIntersectionControlEnv):
    def __init__(self, max_agents: int = 4, *args, **kwargs):
        if "nagents" in kwargs:
            raise Exception(
                "nagents cannot be assigned for"
                + " RoadIntersectionContinuousFlowControlEnv"
            )
        super().__init__(*args, **kwargs)
        self.nagents = 0
        self.agent_ids = []
        self.agents = {a_id: None for a_id in self.agent_ids}
        self.queue1 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.queue2 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.prev_actions = {
            a_id: torch.zeros(2) for a_id in self.get_agent_ids_list()
        }
        self.curr_actions = {a_id: None for a_id in self.get_agent_ids_list()}
        self.max_agents = max_agents

    def configure_action_list(self):
        # For this env I am adding smoother actions
        self.actions_list = [
            torch.as_tensor(ac).unsqueeze(0)
            for ac in itertools.product(
                np.arange(-0.1, 0.101, 0.05), np.arange(-1.5, 1.51, 0.25),
            )
        ]

    def add_vehicle(self, a_id: str, *args, **kwargs):
        if a_id not in self.get_agent_ids_list() and kwargs.get("place", True):
            assert self.nagents < self.max_agents
            self.agent_ids.append(a_id)
            self.agents[a_id] = None
            self.nagents += 1
            self.queue1[a_id] = deque(maxlen=self.history_len)
            self.queue2[a_id] = deque(maxlen=self.history_len)
            self.prev_actions[a_id] = torch.zeros(2)
            self.curr_actions[a_id] = torch.zeros(2)
        return super().add_vehicle(a_id, *args, **kwargs)

    def handle_goal_tolerance(self, agent):
        rew = 0.0
        if (
            agent["vehicle"].distance_from_destination() < self.goal_tolerance
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
                rew = self.goal_reach_bonus
            else:
                rew = -torch.abs(
                    (
                        agent["vehicle"].speed
                        / (agent["dynamics"].v_lim * self.horizon)
                    )
                ).item()

            # Remove the agent which has completed the task
            a_id = agent["vehicle"].name
            srd = int(self.agents[a_id]["road name"][-1])
            # FIXME: No turns for now
            erd = (srd + 2) % 4
            self.world.dynamic_environment(a_id)
            self.nagents -= 1
            self.agent_ids.remove(a_id)
            del self.agents[a_id]
            del self.queue1[a_id]
            del self.queue2[a_id]
            del self.prev_actions[a_id]
            del self.curr_actions[a_id]

            # Add a new agent in its place
            self._add_vehicle_with_collision_check(a_id, srd, erd, self.mode == 2)
        return rew

    def setup_nagents_1(self):
        # Start at the road "traffic_signal_0" as the state space is
        # invariant to rotation and start position
        erd = np.random.choice([2])
        a_id = "agent_0"

        if self.mode == 1:
            # Car starts at the road center
            self.add_vehicle_path(a_id, 0, erd, False)
        elif self.mode == 2:
            self.add_vehicle_path(a_id, 0, erd, True)
        else:
            raise NotImplementedError

    def setup_nagents_2(self, bypass_mode=None, sample=False):
        mode = self.mode if bypass_mode is None else bypass_mode
        sample = True if (mode > 3 or sample) else False
        mode = (mode - 1) % 3 + 1 if mode > 3 else mode

        if mode == 1:
            # Perpendicular cars. Learn only traffic signal
            self.add_vehicle_path("agent_0", 0, 2, sample)

            # Choose which road the next car is placed
            srd = np.random.choice([1, 3])
            self.add_vehicle_path("agent_1", srd, (srd + 2) % 4, sample)
        elif mode == 2:
            # Parallel cars. Learn only lane following
            self.add_vehicle_path("agent_0", 0, 2, sample)
            self.add_vehicle_path("agent_1", 2, 0, sample)
        elif mode == 3:
            # Sample uniformly between modes 1 and 2
            self.setup_nagents_2(
                bypass_mode=np.random.choice([1, 2]), sample=sample
            )

    def reset(self):
        self.world = self.generate_world_without_agents()

        if self.max_agents == 1:
            self.setup_nagents_1()
        elif self.max_agents == 2:
            self.setup_nagents_2()
        else:
            self.setup_nagents(self.max_agents)

        return super(RoadIntersectionControlEnv, self).reset()
