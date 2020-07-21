import itertools
import math
import random
from collections import deque

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple
from sdriving.agents.model import PPOLidarActorCritic
from sdriving.envs.intersection_env import RoadIntersectionControlEnv
from sdriving.trafficsim.common_networks import (
    generate_intersection_world_4signals,
    generate_intersection_world_12signals,
)
from sdriving.trafficsim.dynamics import CatmullRomSplineAccelerationModel
from sdriving.trafficsim.utils import (
    angle_normalize,
    get_2d_rotation_matrix,
    transform_2d_coordinates_rotation_matrix,
    invtransform_2d_coordinates_rotation_matrix,
)
from sdriving.trafficsim.vehicle import Vehicle
from sdriving.trafficsim.world import World


class RoadIntersectionDualObjective(RoadIntersectionControlEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.controller_action_space = self.get_controller_action_space()
        self.controller_observation_space = (
            self.get_controller_observation_space()
        )

        self.spline_action_space = self.get_spline_action_space()
        self.spline_observation_space = self.get_spline_observation_space()

    def get_controller_action_space(self):
        self.max_accln = 1.5
        self.controller_actions_list = torch.linspace(
            -self.max_accln, self.max_accln, steps=10
        ).unsqueeze(0)
        return Discrete(self.controller_actions_list.size(1))

    def get_controller_observation_space(self):
        return super().get_observation_space()

    def get_spline_observation_space(self):
        return Box(
            low=np.array([-1.0, -1.0] * 3), high=np.array([-1.0, -1.0] * 3),
        )

    def get_spline_action_space(self):
        # TODO: Once this works experiment with more deviation patterns
        self.nwaypoints = 1
        return Box(
            low=np.array([0.0, -1.0] * self.nwaypoints),
            high=np.array([1.0, 1.0] * self.nwaypoints),
        )

    def add_vehicle(
        self,
        a_id,
        rname,
        pos,
        v_lim,
        orientation,
        dest,
        dest_orientation,
        dynamics_model=CatmullRomSplineAccelerationModel,
        dynamics_kwargs={},
    ):
        ret_val = super().add_vehicle(
            a_id,
            rname,
            pos,
            v_lim,
            orientation,
            dest,
            dest_orientation,
            CatmullRomSplineAccelerationModel,
            dynamics_kwargs,
        )
        self.agents[a_id]["track_point"] = 0
        self.agents[a_id]["previous_start_point"] = self.agents[a_id][
            "vehicle"
        ].position
        # Add a dummy start point
        self.agents[a_id]["track"] = [
            self.get_dummy_point(self.agents[a_id]["previous_start_point"]),
            self.agents[a_id]["previous_start_point"].unsqueeze(0),
        ]
        return ret_val

    def get_dummy_point(self, pt: torch.Tensor):
        # This point lies in one of the roads not in a gray area
        x, y = pt
        if x > self.width / 2:
            return torch.as_tensor(
                [self.length + self.width / 2, y]
            ).unsqueeze(0)
        elif x < -self.width / 2:
            return torch.as_tensor(
                [-(self.length + self.width / 2), y]
            ).unsqueeze(0)
        elif y > self.width / 2:
            return torch.as_tensor(
                [x, self.length + self.width / 2]
            ).unsqueeze(0)
        else:  # if y < -self.width / 2:
            return torch.as_tensor(
                [x, -(self.length + self.width / 2)]
            ).unsqueeze(0)

    def get_controller_state(self):
        self.prev_states = {
            a_id: self.get_controller_state_single_agent(a_id)
            for a_id in self.get_agent_ids_list()
        }
        return self.prev_states

    def get_spline_state(self):
        return {
            a_id: self.get_spline_state_single_agent(a_id)
            for a_id in self.get_agent_ids_list()
        }

    def get_controller_state_single_agent(self, a_id: str):
        return super().get_state_single_agent(a_id)

    def get_spline_state_single_agent(self, a_id: str):
        lw = 2 * (self.length + self.width / 2)
        pts = []
        for pt in self.agents[a_id]["intermediate_goals"][:-1]:
            pts.append(pt[:2].clone())

        rot_mat = get_2d_rotation_matrix(
            self.agents[a_id]["vehicle"].orientation
        )
        offset = self.agents[a_id]["vehicle"].position
        pts = torch.cat(pts + [self.agents[a_id]["vehicle"].destination])

        pts = transform_2d_coordinates_rotation_matrix(
            pts.reshape(-1, 2), rot_mat, offset
        ).reshape(-1)

        self.agents[a_id]["shortest_path_points"] = pts
        self.agents[a_id]["transformation"] = [rot_mat, offset]
        return pts / lw

    def transform_state_action(self, actions, states, timesteps):
        nactions = {}
        nstates = {}
        extras = {}
        for a_id in self.get_agent_ids_list():
            self.check_in_space(self.controller_action_space, actions[a_id])
            self.check_in_space(
                self.controller_observation_space, states[a_id]
            )
            # actions --> Goal State for MPC
            # states  --> Start State for MPC
            # extras  --> None if using MPC, else tuple of
            #             nominal states, actions
            (
                nactions[a_id],
                nstates[a_id],
                extras[a_id],
            ) = self.transform_state_action_single_agent(
                a_id, actions[a_id], states[a_id], timesteps
            )
        return nactions, nstates, extras

    def transform_state_action_single_agent(
        self, a_id: str, action: torch.Tensor, state, timesteps: int
    ):
        agent = self.agents[a_id]["vehicle"]

        x, y = agent.position
        v = agent.speed
        t = agent.orientation

        start_state = torch.as_tensor([x, y, v, t])
        action = self.controller_actions_list[:, action].unsqueeze(1)
        dynamics = self.agents[a_id]["dynamics"]
        nominal_states = [start_state.unsqueeze(0)]
        nominal_actions = [action]

        for _ in range(timesteps):
            start_state = nominal_states[-1]
            new_state = dynamics(start_state, action)
            nominal_states.append(new_state.cpu())
            nominal_actions.append(action)

        nominal_states, nominal_actions = (
            torch.cat(nominal_states),
            torch.cat(nominal_actions),
        )
        na = torch.zeros(4)
        ns = torch.zeros(4)
        ex = (nominal_states, nominal_actions)

        self.curr_actions[a_id] = action[0]

        return na, ns, ex

    def spline_step(self, actions: dict):
        for a_id, action in actions.items():
            # deviations = action
            pts = self.agents[a_id]["shortest_path_points"]
            deviations = torch.zeros_like(pts)
            for i in range(pts.size(0) // 2):
                # TODO: Predict Generic Waypoints
                r = action[0]  # 2 * i]
                theta = action[1]  # 2 * i + 1]
                deviations[2 * i] = (
                    r * self.width * torch.cos(math.pi * theta) / 2
                    + pts[2 * i]
                )
                deviations[2 * i + 1] = (
                    r * self.width * torch.sin(math.pi * theta) / 2
                    + pts[2 * i + 1]
                )
            track = invtransform_2d_coordinates_rotation_matrix(
                deviations.reshape(-1, 2), *self.agents[a_id]["transformation"]
            )
            self.agents[a_id]["track"].append(track)
            self.agents[a_id]["track"].append(
                self.get_dummy_point(self.agents[a_id]["track"][-1][-1])
            )
            track = torch.cat(self.agents[a_id]["track"], dim=0)
            self.agents[a_id]["dynamics"].register_track(
                track, dummy_point=True
            )
        return None

    def controller_step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def post_process_rewards(self, rewards, now_dones):
        # Encourage the agents to make smoother transitions
        for a_id in self.get_agent_ids_list():
            if a_id not in rewards:
                continue
            rew = rewards[a_id]
            pac = self.prev_actions[a_id]
            if now_dones is None:
                # The penalty should be lasting for all the
                # timesteps the action was taken. None is passed
                # when the intermediate timesteps have been
                # processed, so swap the actions here
                self.curr_actions[a_id] = pac
                return
            cac = self.curr_actions[a_id]
            if a_id not in self.prev_actions or a_id not in self.curr_actions:
                # Agents are removed in case of Continuous Flow Environments
                continue
            diff = torch.abs(pac - cac)
            penalty = diff[0] / (2 * self.max_accln * self.horizon)
            rewards[a_id] = rew - penalty

    def render(self, *args, **kwargs):
        pts = {}
        for a_id in self.get_agent_ids_list():
            pts[a_id] = []
            track = self.agents[a_id]["track"]
            for pt in track:
                for i in range(pt.size(0)):
                    pts[a_id].append(pt[i].detach().cpu().numpy())
            pts[a_id] = pts[a_id][1:-1]
        kwargs.update({"pts": pts})
        super().render(*args, **kwargs)
