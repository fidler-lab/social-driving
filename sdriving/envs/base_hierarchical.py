from itertools import combinations
from typing import Optional

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from celluloid import Camera
from sdriving.agents.model import ActiveSplineTorch
from sdriving.envs.base_env import BaseEnv
from sdriving.trafficsim.utils import check_intersection_lines
from sdriving.trafficsim.world import World

matplotlib.use("Agg")

plt.style.use("seaborn-pastel")


class BaseHierarchicalEnv(BaseEnv):
    def __init__(
        self,
        world: World,
        nagents: int,
        horizon: Optional[int] = None,
        tolerance: float = 4.0,
        object_collision_penalty: float = 1.0,
        agents_collision_penalty: float = 1.0,
        cp_num: int = 6,
        p_num: int = 500,
    ):
        self.cp_num = cp_num
        self.p_num = p_num
        # Need to figure out how to do this properly as the subclasses use
        # Multiple Inheritance
        BaseEnv.__init__(
            self,
            world,
            nagents,
            horizon,
            tolerance,
            object_collision_penalty,
            agents_collision_penalty,
            True,
        )

        self.spline = ActiveSplineTorch(cp_num, p_num)
        self.metacontrol_cache = {
            a_id: False for a_id in self.get_agent_ids_list()
        }
        self.control_points = dict()
        self.points = dict()

        self.extrinsic_reward = dict()

        self.observation_space = {
            "MetaController": self._get_meta_controller_observation_space(),
            "Controller": self._get_controller_observation_space(),
        }
        self.action_space = {
            "MetaController": self._get_meta_controller_action_space(),
            "Controller": self._get_controller_action_space(),
        }

    def get_action_space(self):
        return None

    def get_observation_space(self):
        return None

    def configure_action_list(self):
        self._configure_metacontroller_actions()
        self._configure_controller_actions()

    def transform_state_action(
        self, actions: dict, states: dict, timesteps: int
    ):
        nactions = dict()
        nstates = dict()
        extras = dict()
        for a_id in self.get_agent_ids_list():
            action_numpy = self.convert_to_numpy(actions[a_id])
            state_numpy = self.convert_to_numpy(states[a_id])

            assert self.action_space["Controller"].contains(
                action_numpy
            ), f"{action_numpy} doesn't lie in {self.action_space['Controller']}"
            assert self.observation_space["Controller"].contains(
                state_numpy
            ), f"{state_numpy} doesn't lie in {self.observation_space['Controller']}"

            (
                nactions[a_id],
                nstates[a_id],
                extras[a_id],
            ) = self.transform_state_action_single_agent(
                a_id, actions[a_id], states[a_id], timesteps
            )
        return nactions, nstates, extras

    def step(
        self,
        actions: dict,
        metacontrol: bool,
        timesteps: int = 10,
        render: bool = False,
        tolerance: float = 4.0,
        **kwargs,
    ):
        if metacontrol:
            return self._step_metacontrol(actions)
        else:
            return self._step_control(
                actions,
                timesteps=timesteps,
                render=render,
                tolerance=tolerance,
                **kwargs,
            )

    def _step_metacontrol(self, actions):
        for a_id, idx in actions.items():
            cps = [self.agents[a_id]["vehicle"].position.unsqueeze(0)]
            cps.extend(self.meta_controller_actions[idx])
            # cps.append(self.agents[a_id]["original_destination"].unsqueeze(0))
            cps = torch.cat(cps)
            # cps should include the starting and ending positions
            self.control_points[a_id] = cps
            self.points[a_id] = self.spline(cps.unsqueeze(0)).squeeze(0)
            self.metacontrol_cache[a_id] = True
            self.agents[a_id]["dynamics"].register(self.points[a_id])

    def _step_control(self, actions, timesteps, render, tolerance, **kwargs):
        assert all(
            self.metacontrol_cache.values()
        ), "Control Points need to be set"
        return super().step(actions, timesteps, render, tolerance, **kwargs)

    def update_extrinsic_reward(self):
        rewards = super().get_reward()
        for a_id, reward in rewards.items():
            self.extrinsic_reward[a_id] += reward

    def get_reward(self):
        self.update_extrinsic_reward()
        return self.get_intrinsic_reward()

    def get_intrinsic_reward(self):
        # TODO: Figure out the exact reward for this
        return {a_id: 0.0 for a_id in self.get_agent_ids_list()}

    def get_extrinsic_reward(self):
        return self.extrinsic_reward

    def get_state(self, metacontrol: bool = False):
        if metacontrol:
            return {
                a_id: self.get_state_single_agent_metacontroller(a_id)
                for a_id in self.get_agent_ids_list()
            }
        else:
            return super().get_state()

    def reset(self):
        self.metacontrol_cache = {
            a_id: False for a_id in self.get_agent_ids_list()
        }
        self.control_points = dict()
        self.points = dict()
        self.extrinsic_reward = {
            a_id: 0.0 for a_id in self.get_agent_ids_list()
        }
        return super().reset()
