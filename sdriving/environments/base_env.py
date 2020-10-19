from collections import OrderedDict
from copy import copy
from typing import Union

import numpy as np
import torch

from sdriving.tsim.world import World


class BaseMultiAgentDrivingEnvironment:
    def __init__(
        self,
        world: World,
        nagents: int,
        horizon: int = np.inf,
        timesteps: int = 25,
        device: torch.device = torch.device("cpu"),
    ):
        # Get the action and observation space
        self.configure_action_space()
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.timesteps = timesteps
        self.cached_actions = None
        # This needs to be set before calling `step`
        self.dynamics = None
        self.collision_vector = torch.zeros(nagents, 1).bool()
        self.completion_vector = torch.zeros(nagents, 1).bool()

        # Ensure that the road network has its network configured
        self.world = world

        # Setup agent ids
        self.nagents = nagents
        self.actual_nagents = nagents
        self.agent_ids = [f"agent"]  # All agents grouped into 1 vehicle
        self.agent_names = [
            f"agent_{i}" for i in range(self.nagents)
        ]  # Right now we support removal of agents.
        # This list is the actual set of agents present in the environment.
        self.agent_names_copy = [f"agent_{i}" for i in range(self.nagents)]
        self.agents = OrderedDict()

        # Stats
        self.nsteps = 0
        self.nepisodes = 0
        self.horizon = horizon

        self.device = device

    def register_reward(self, reward):
        # Needed to implement curriculum strategies
        pass

    def sync(self):
        # Sync properties across different processes
        pass

    def to(self, device: torch.device):
        self.transfer_dict(self.__dict__, device)
        self.device = device

    def transfer_dict(self, d: Union[dict, OrderedDict], device: torch.device):
        for k, t in d.items():
            if torch.is_tensor(t):
                d[k] = t.to(device)
            elif hasattr(t, "to"):
                t.to(device)
            elif isinstance(t, (dict, OrderedDict)):
                self.transfer_dict(t, device)

    def configure_action_space(self):
        # Needed if you are using a discrete action space
        pass

    def reset(self):
        self.nagents = self.actual_nagents
        self.collision_vector = torch.zeros(self.nagents, 1).bool()
        self.completion_vector = torch.zeros(self.nagents, 1).bool()
        self.nsteps = 0
        self.nepisodes += 1
        self.agent_names = [f"agent_{i}" for i in range(self.nagents)]
        self.agent_names_copy = [f"agent_{i}" for i in range(self.nagents)]
        self.to(self.device)
        return self.get_state()

    def get_agent_ids_list(self):
        return self.agent_ids

    def check_in_space(self, space, val):
        val = self.convert_to_numpy(val)
        assert space.contains(val), f"{val} doesn't lie in {space}"

    def assert_in_action_space(self, val):
        self.check_in_space(self.action_space, val)

    def assert_in_observation_space(self, val):
        self.check_in_space(self.observation_space, val)

    @staticmethod
    def convert_to_numpy(tensor):
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            return [t.detach().cpu().numpy() for t in tensor]
        return tensor.detach().cpu().numpy()

    def discrete_to_continuous_actions(self, action: torch.Tensor):
        return action

    def get_state(self):
        raise NotImplementedError

    def get_reward(self, new_collisions: torch.Tensor, action: torch.Tensor):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        self.world.render(*args, **kwargs)

    def vehicle_collision_check(self, v):
        # This function is needed when experimenting with lateral deviation noise
        return v.collision_check()

    @torch.no_grad()
    def step(
        self, action: torch.Tensor, render: bool = False, **render_kwargs
    ):
        action = self.discrete_to_continuous_actions(action)
        action = action.to(self.world.device)
        accumulated_reward = torch.zeros(
            action.size(0), 1, device=self.world.device
        )

        for t in range(self.timesteps):
            prev_state = self.world.get_all_vehicle_state()
            state = self.dynamics(prev_state, action)
            state = (
                ~self.collision_vector * state
                + self.collision_vector * prev_state
            )
            i = 0
            collision_vehicle, collision_object = [], []
            for n, v in self.world.vehicles.items():
                self.world.update_state(
                    n, state[i : (i + v.nbatch)], wait=t < self.timesteps - 1
                )
                i += v.nbatch
                # TODO: Cross Vehicle Collision (should ideally be avoided
                #       by having only one fleet of vehicles)
                collision_vehicle.append(self.vehicle_collision_check(v))
                collision_object.append(self.world.check_collision(n))
            collision_vehicle = torch.cat(collision_vehicle)
            collision_object = torch.cat(collision_object)
            new_collision = collision_vehicle + collision_object
            new_collision = new_collision.unsqueeze(1)

            if render:
                self.render(**render_kwargs)

            rew = self.get_reward(new_collision, action)
            accumulated_reward += rew
            if self.collision_vector.all() or self.horizon <= self.nsteps:
                break
            self.nsteps += 1
            self.world.update_world_state(1)

        if hasattr(self, "remove"):
            idxs = torch.where(self.completion_vector)[0]
            agent_names_copy = copy(self.agent_names)
            for i in idxs:
                self.remove(agent_names_copy[i])

        self.cached_actions = action

        timeout = self.horizon <= self.nsteps

        dones = self.collision_vector + timeout + (len(self.agent_names) == 0)

        return (
            self.get_state() if not dones.all() else (None, None),
            accumulated_reward,
            dones,
            {"timeout": timeout},
        )
