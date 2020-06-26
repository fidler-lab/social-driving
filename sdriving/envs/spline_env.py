import math

import sdriving
from sdriving.envs.intersection_env import RoadIntersectionControlEnv
from sdriving.trafficsim.dynamics import ParametricBicycleKinematicsModel
from sdriving.trafficsim.parametric_curves import CatmullRomSplineMotion
from sdriving.trafficsim.utils import circle_segment_area

from spinup.utils.mpi_tools import proc_id

import torch
from torch import nn

import numpy as np


class RoadIntersectionSplineControlEnv(RoadIntersectionControlEnv):
    # This env has 2 objectives. One agent needs to solve a differentiable
    # objective for proper pathing. Another agent needs to control the speed
    # of the agent along that path
    def configure_action_list(self):
        self.actions_list = [
            torch.as_tensor([[ac]]) for ac in np.arange(-1.5, 1.75, 0.25)
        ]
        self.max_accln = 1.5

    def get_starting_positions(self, as_tensor: bool = False):
        pos = {
            a_id: self.agents[a_id]["vehicle"].position.clone().unsqueeze(0)
            for a_id in self.get_agent_ids_list()
        }
        if as_tensor:
            return torch.cat(list(pos.values()), dim=0)
        return pos

    def get_intermediate_goals(self, as_tensor: bool = False):
        goals = {
            a_id: torch.cat(
                [
                    tensor[:2].unsqueeze(0).unsqueeze(0)
                    for tensor in self.agents[a_id]["intermediate_goals"]
                ],
                dim=1,
            )
            for a_id in self.get_agent_ids_list()
        }
        if as_tensor:
            return torch.cat(list(goals.values()), dim=0)
        return goals

    def register_dynamics_track(
        self, tracks: dict, motion: torch.nn.Module = CatmullRomSplineMotion
    ):
        for a_id in self.get_agent_ids_list():
            self.agents[a_id]["dynamics"] = ParametricBicycleKinematicsModel(
                tracks[a_id], motion, model_kwargs={"p_num": 50, "alpha": 10.0}
            )

    def register_track(self, model: nn.Module):
        track = model(
            self.get_starting_positions(True),
            self.get_intermediate_goals(True),
            self.length,
            self.width,
        )
        self.register_dynamics_track(
            {
                a_id: track[i, :, :]
                for (i, a_id) in enumerate(self.get_agent_ids_list())
            }
        )

    def get_agent_driving_state(self, a_id: str):
        agent = self.agents[a_id]["vehicle"]
        x, y = agent.position
        v = agent.speed
        t = agent.orientation

        return torch.as_tensor([x, y, v, t])

    def compute_loss(
        self,
        a_id: str,
        state: torch.Tensor,
        action: torch.Tensor,
        prev_state: torch.Tensor,
    ):
        dest = self.agents[a_id]["original_destination"]
        dist = (state[:2] - dest).pow(2).sum().sqrt()
        theta_diff = torch.abs(state[3] - prev_state[3]) / (2 * math.pi)
        collision = (
            torch.mean(self.smooth_collision_penalty(a_id))
            / self.agents[a_id]["vehicle"].area
        )

        return (
            5.0
            * dist
            / ((self.agents[a_id]["original_distance"] + 1e-12) * self.horizon)
            + 1.0 * theta_diff / self.horizon
            + 100.0
            * collision
            / self.horizon  # Additional weightage on collision
        )

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
            penalty = (diff[0] / (2 * self.max_accln)) / self.horizon
            rewards[a_id] = rew - penalty

    def step(
        self,
        actions: dict,
        render: bool = False,
        tolerance: float = 4.0,
        differentiable_objective: bool = False,
        **kwargs,
    ):
        self.update_env_state()
        states = self.prev_state
        # The actions and states are ideally in global frame
        actions, states, extras = self.transform_state_action(
            actions, states, 10
        )

        nstates = {a_id: extras[a_id][0] for a_id in extras.keys()}
        nactions = {a_id: extras[a_id][1] for a_id in extras.keys()}

        timesteps = 10

        id_list = self.get_agent_ids_list()

        losses = {a_id: 0.0 for a_id in id_list}
        rewards = {a_id: 0.0 for a_id in id_list}

        prev_states = {
            a_id: self.get_agent_driving_state(a_id) for a_id in id_list
        }

        for i in range(timesteps):
            for a_id in id_list:
                if not self.is_agent_done(a_id):
                    state = nstates[a_id][i, :]
                    if self.astar_nav and self.agents[a_id][
                        "prev_point"
                    ] < len(self.agents[a_id]["intermediate_goals"]):
                        xg, yg, vg, _ = self.agents[a_id][
                            "intermediate_goals"
                        ][self.agents[a_id]["prev_point"]]
                        dest = torch.as_tensor([xg, yg])
                        if (
                            self.world.road_network.is_perpendicular(
                                self.world.vehicles[a_id].road,
                                dest,
                                state[:2],
                            )
                            or ((dest - state[:2]) ** 2).sum().sqrt()
                            < self.goal_tolerance
                        ):
                            self.agents[a_id]["prev_point"] += 1
                    self.world.update_state(
                        a_id,
                        state,
                        change_road_association=(i == timesteps - 1),
                    )

                    # Compute a penalty
                    if differentiable_objective:
                        if self.world.check_collision(a_id):
                            self.agents[a_id]["done"] = True

                        losses[a_id] += self.compute_loss(
                            a_id,
                            state,
                            nactions[a_id][i, :],
                            prev_states[a_id],
                        )

                    prev_states[a_id] = state
            if not differentiable_objective:
                intermediate_rewards = self.get_reward()
                for a_id in id_list:
                    if a_id not in rewards or a_id not in intermediate_rewards:
                        continue
                    rewards[a_id] += intermediate_rewards[a_id]
            if render:
                self.world.render(**kwargs)

        self.nsteps += 10
        self.world.update_world_state(i + 1)

        if not differentiable_objective:
            self.post_process_rewards(rewards, None)

        return (
            self.get_state(),
            losses if differentiable_objective else rewards,
            self.is_done(),
            {"timeout": self.horizon < self.nsteps},
        )

    def _get_distances(
        self, pt: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor
    ):
        if pt.ndim == 1:
            pt = pt.unsqueeze(0)
        l2 = (pt1 - pt2).pow(2).sum(1)
        t = ((pt - pt1).matmul((pt2 - pt1).T)).diag() / l2
        t = torch.max(
            torch.zeros_like(t), torch.min(torch.ones_like(t), t)
        ).unsqueeze(1)
        projection = pt1 + t * (pt2 - pt1)
        distance = (pt - projection).pow(2).sum(1).sqrt()
        return distance

    def smooth_collision_penalty(self, a_id: str, pos=None):
        vehicle = self.agents[a_id]["vehicle"]
        ventity = self.world.vehicles[a_id]
        pt1, pt2 = self.world.road_network.get_neighbouring_edges(
            ventity.road,
            vname=a_id,
            type="road" if not ventity.grayarea else "garea",
            cars=False,
        )
        distance = self._get_distances(
            vehicle.position if pos is None else pos, pt1, pt2
        )

        return circle_segment_area(distance, vehicle.safety_circle)
