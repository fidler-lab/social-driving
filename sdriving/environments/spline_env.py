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
    SplineModel,
    get_2d_rotation_matrix,
)
from sdriving.agents.model import PPOLidarActorCritic


class MultiAgentOneShotSplinePredictionEnvironment(
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment
):
    def __init__(self, acceleration_agent: str, *args, **kwargs):
        """
        `acceleration_agent`: Must be the path to a saved pretrained agent.
                              Should have been trained on the
                              MultiAgentIntersectionFixedTrackEnvironment.
                              Also, the action space is assumed to be
                              discrete.
        """
        super().__init__(*args, **kwargs)

        ckpt = torch.load(acceleration_agent, map_location=self.device)
        centralized = ckpt["model"] == "centralized_critic"
        self.accln_control = PPOLidarActorCritic(
            **ckpt["ac_kwargs"], centralized=centralized
        ).eval()
        self.accln_control.v = None
        self.accln_control.pi.load_state_dict(ckpt["actor"])
        self.accln_control_actions_list = torch.arange(-1.5, 1.55, step=0.25)
        self.accln_control_actions_list.unsqueeze_(1)

    def get_observation_space(self):
        self.nwaypoints = 3
        return Box(
            low=np.array([-1.0] * 2 * self.nwaypoints),
            high=np.array([1.0] * 2 * self.nwaypoints),
        )

    def get_action_space(self):
        return Box(
            low=np.array([0.0] * 2 * self.nwaypoints),
            high=np.array([1.0, 2 * math.pi] * self.nwaypoints),
        )

    def get_state(self):
        if self.got_spline_state:
            # When the base `step` method is called it will keep
            # invoking `get_state` function. This check allows us
            # to invoke the `get_state` of the superclass
            return super().get_state()
        self.got_spline_state = True
        a_ids = self.get_agent_ids_list()
        # The points we receive from the world are in the global
        # frame. To make learning easier we need to transform them
        # to the local car frame.
        vehicle = self.agents["agent"]
        destinations = vehicle.destination.unsqueeze(1)  # N x 1 x 2
        feasible_path = self.world.trajectory_points["agent"]  # N x B x 2
        feasible_path = torch.cat(
            [feasible_path, destinations], dim=1
        )  # N x (B + 1) x 2
        lw = 2 * self.length + self.width
        rot_mat = get_2d_rotation_matrix(vehicle.orientation).permute(
            0, 2, 1
        )  # N x 2 x 2
        offset = vehicle.position.unsqueeze(1)  # N x 1 x 2
        self.transformation = (rot_mat, offset)

        local_feasible_path = torch.bmm(
            feasible_path - offset, rot_mat  # N x (B + 1) x 2
        )
        self.cached_path = local_feasible_path

        return local_feasible_path.view(self.nagents, -1) / lw

    def get_reward(self, new_collisions: torch.Tensor, action: torch.Tensor):
        a_ids = self.get_agent_ids_list()

        # Distance from destination
        distances = torch.cat(
            [self.agents[v].distance_from_destination() for v in a_ids]
        )

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
        penalty = (
            new_collisions.float()
            + new_collisions
            * distances
            * (self.horizon - self.nsteps - 1)
            / self.horizon
        )
        self.collision_vector += new_collisions

        return (
            -distances * (~self.collision_vector) / self.horizon
            - penalty
            + goal_reach_bonus
        )

    def store_dynamics(self, vehicle):
        self.dynamics = None

    def reset(self):
        self.got_spline_state = False
        return super().reset()

    @torch.no_grad()
    def step(
        self, action: torch.Tensor, render: bool = False, **render_kwargs
    ):
        action = self.discrete_to_continuous_actions(action)
        action = action.to(self.world.device)

        rot_mat, offset = self.transformation
        action = action.view(self.nagents, -1, 2)
        radii = action[..., 0:1] * self.width / 2
        theta = action[..., 1:2] * math.pi
        del_x = torch.cos(theta) * radii
        del_y = torch.sin(theta) * radii
        path = self.cached_path + torch.cat([del_x, del_y], dim=-1)
        action = torch.baddmm(offset, path, torch.inverse(rot_mat))

        self.dynamics = SplineModel(
            action, v_lim=torch.ones(self.nagents) * 8.0
        )

        accumulated_reward = torch.zeros(
            action.size(0), 1, device=self.world.device
        )

        done = False
        obs = self.get_state()
        while not done:
            action = self.accln_control(obs, deterministic=True)
            obs, reward, dones, _ = super().step(
                action, render, **render_kwargs
            )
            accumulated_reward += reward
            done = dones.all()

        return None, accumulated_reward, None, None
