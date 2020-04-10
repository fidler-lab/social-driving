import itertools
import math

import numpy as np
import torch

from sdriving.trafficsim.dynamics import (
    BicycleKinematicsModel as VehicleDynamics,
)
from sdriving.trafficsim.mpc import MPCController
from sdriving.trafficsim.vehicle import Vehicle
from sdriving.trafficsim.world import World


class MPCControllerWrapper:
    def __init__(
        self,
        world: World,
        timesteps_meta_controller: int = 10,
        tolerance: float = 0.0,
        lqr_iter: int = 5,
        **kwargs,
    ):
        self.mpc = MPCController(**kwargs)
        self.timesteps_meta_controller = timesteps_meta_controller
        self.lqr_iter = lqr_iter
        self.world = world
        self.tolerance = -np.inf if tolerance == 0.0 else tolerance

        self.device = torch.device("cpu")

    def to(self, device):
        if device == self.device:
            return
        self.device = device
        self.mpc.to(device)

    def __call__(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        vehicle: Vehicle,
        dynamics: VehicleDynamics,
        info: dict,
    ):
        dynamics.to(self.device)

        nominal_states, nominal_actions, _ = self.mpc(
            start_state.to(self.device),
            goal_state.to(self.device),
            dynamics,
            timesteps=self.timesteps_meta_controller,
            lqr_iter=self.lqr_iter,
        )
        nominal_states = nominal_states[:, 0, :]
        nominal_actions = nominal_actions[:, 0, :]
        return nominal_states, nominal_actions, None


class HybridController:
    def __init__(
        self,
        world: World,
        agent: str,
        timesteps_meta_controller: int = 10,
        tolerance: float = 0.0,
        npoints: int = 100,
        **kwargs,
    ):
        self.world = world
        self.timesteps_meta_controller = timesteps_meta_controller
        self.tolerance = -np.inf if tolerance == 0.0 else tolerance

        ckpt = torch.load(agent, map_location="cpu")
        if ckpt["model"] == "centralized_critic":
            from sdriving.agents.ppo_cent.model import ActorCritic
        self.agent = ActorCritic(**ckpt["ac_kwargs"])
        self.agent.v = None
        self.agent.pi.load_state_dict(ckpt["actor"])

        self.npoints = npoints
        self.actions_list = [
            torch.as_tensor(ac)
            for ac in itertools.product(
                [-0.1, -0.05, 0.0, 0.05, 0.1], [-1.5, -0.75, 0.0, 0.75, 1.5],
            )
        ]

        self.device = torch.device("cpu")

    def to(self, device):
        if self.device == device:
            return
        self.device = device
        self.agent.to(device)
        self.actions_list = [act.to(device) for act in self.actions_list]

    def get_agent_state(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        vehicle: Vehicle,
        v_lim: float,
    ):
        dest = goal_state[:2]
        pos = start_state[:2]
        inv_dist = 1 / ((pos - dest) ** 2).sum().sqrt()
        lidar_data = 1 / self.world.get_lidar_data_from_state(
            start_state, vehicle.name, self.npoints
        )
        return (
            torch.as_tensor(
                [
                    (goal_state[2] - start_state[2]) / (2 * v_lim),
                    vehicle.optimal_heading_to_point(dest) / math.pi,
                    inv_dist if torch.isfinite(inv_dist) else 0.0,
                ]
            ).to(self.device),
            lidar_data.to(self.device),
        )

    def __call__(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        vehicle: Vehicle,
        dynamics: VehicleDynamics,
        info: dict,
    ):
        state = self.get_agent_state(
            start_state, goal_state, vehicle, info["v_lim"]
        )
        act = self.agent.act(state, deterministic=True).item()

        if act > 0:
            action = self.actions_list[act - 1].unsqueeze(0)
            nominal_states = [start_state.unsqueeze(0)]
            nominal_actions = [action]
            for _ in range(self.timesteps_meta_controller):
                start_state = nominal_states[-1]
                new_state = dynamics(start_state, action)
                nominal_states.append(new_state.cpu())
                nominal_actions.append(action)
            nominal_states = torch.cat(nominal_states)
            nominal_actions = torch.cat(nominal_actions)
            return (
                torch.zeros(4).to(self.device),
                torch.zeros(4).to(self.device),
                (nominal_states, nominal_actions),
            )
        else:
            return goal_state, start_state, None
