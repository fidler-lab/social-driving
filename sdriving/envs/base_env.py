from itertools import combinations
from typing import Optional

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from celluloid import Camera
from sdriving.trafficsim.utils import check_intersection_lines

matplotlib.use("Agg")

plt.style.use("seaborn-pastel")


class BaseEnv:
    def __init__(
        self,
        world,
        nagents: int,
        horizon: Optional[int] = None,
        tolerance: float = 0.0,
        object_collision_penalty: float = 1.0,
        agents_collision_penalty: float = 1.0,
        astar_nav: bool = False,
    ):
        # Get the action and observation space
        self.configure_action_list()
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # Configure the world
        self.world = world
        self.world.road_network.construct_graph()

        # Setup agent ids
        self.nagents = nagents
        self.agent_ids = [f"agent_{i}" for i in range(nagents)]
        self.agents = {a_id: None for a_id in self.agent_ids}

        # Object and Agent Collision
        self.object_collision_penalty = object_collision_penalty
        self.agents_collision_penalty = agents_collision_penalty

        # Navigation Utilities
        self.goal_tolerance = tolerance
        self.astar_nav = astar_nav

        # Stats
        self.nsteps = 0
        self.nepisodes = 0
        self.horizon = horizon

        # Rendering utilities
        self.fig = None
        self.cam = None
        self.ax = None

    def get_agent_ids_list(self):
        return self.agent_ids

    def check_in_space(self, space, val):
        val = self.convert_to_numpy(val)
        assert space.contains(val), f"{val} doesn't lie in {space}"

    @staticmethod
    def convert_to_numpy(tensor):
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            return [t.detach().cpu().numpy() for t in tensor]
        return tensor.detach().cpu().numpy()

    def transform_state_action_single_agent(
        self,
        a_id: str,
        action: torch.Tensor,
        state: torch.Tensor,
        timesteps: int,
    ):
        raise NotImplementedError

    def transform_state_action(self, actions, states, timesteps):
        nactions = {}
        nstates = {}
        extras = {}
        for id in self.get_agent_ids_list():
            self.check_in_space(self.action_space, actions[id])
            self.check_in_space(self.observation_space, states[id])
            # actions --> Goal State for MPC
            # states  --> Start State for MPC
            # extras  --> None if using MPC, else tuple of
            #             nominal states, actions
            (
                nactions[id],
                nstates[id],
                extras[id],
            ) = self.transform_state_action_single_agent(
                id, actions[id], states[id], timesteps
            )
        return nactions, nstates, extras

    def is_agent_done(self, a_id):
        return self.agents[a_id]["done"]

    def update_env_state(self):
        pass

    def get_state_single_agent(self, a_id: str):
        raise NotImplementedError

    def get_state(self):
        agent_ids = self.get_agent_ids_list()
        states = {}
        for id in agent_ids:
            states[id] = self.get_state_single_agent(id)
        self.prev_state = states
        return self.prev_state

    def __repr__(self):
        ret = f"Environment: {self.__class__.__name__}\n"
        for key, value in self.__dict__.items():
            ret += f"\t{key}: {value}\n"
        return ret

    def construct_collision_matrix(self):
        col_matrix_prev = {}
        if hasattr(self, "col_matrix") and self.col_matrix is not None:
            col_matrix_prev = self.col_matrix
        col_matrix = {}
        for key in combinations(self.get_agent_ids_list(), 2):
            if key in col_matrix_prev:
                col_matrix[key] = col_matrix_prev[key]
            else:
                col_matrix[key] = False
        return col_matrix

    def distance_reward_function(self, agent):
        return agent["vehicle"].distance_from_destination() / (
            agent["original_distance"] + 1e-8
        )

    def handle_goal_tolerance(self, agent):
        return 0.0

    def get_reward_env_interaction(self, a_id):
        reward = 0.0
        agent = self.agents[a_id]

        if agent["done"]:
            return reward, False

        reward -= self.distance_reward_function(agent)

        if self.world.check_collision(a_id):
            agent["done"] = True
            reward -= self.object_collision_penalty
            return reward, True

        reward += self.handle_goal_tolerance(agent)

        return reward, False

    def post_process_rewards(self, rewards, now_dones):
        pass

    def get_reward(self):
        # Now done is helpful for sparse reward functions
        rewards, now_done = {}, {}
        for a_id in self.get_agent_ids_list():
            rewards[a_id], now_done[a_id] = self.get_reward_env_interaction(
                a_id
            )

        for key in combinations(self.get_agent_ids_list(), 2):
            if key not in self.col_matrix:
                key = (key[1], key[0])
            if not self.col_matrix[key]:
                self.intervehicle_collision(
                    key, rewards, now_done, self.agents_collision_penalty
                )
        self.post_process_rewards(rewards, now_done)
        return rewards

    def reset(self):
        self.col_matrix = None
        self.col_matrix = self.construct_collision_matrix()
        self.nsteps = 0
        self.nepisodes += 1
        self.prev_state = None
        self.fig = None
        self.cam = None
        self.ax = None
        return self.get_state()

    def intervehicle_collision(self, key, rewards, now_done, penalty):
        id_list = self.get_agent_ids_list()
        id1 = key[0]
        id2 = key[1]
        agent1 = self.agents[id1]
        agent2 = self.agents[id2]
        overlap = agent1["vehicle"].safety_circle_overlap(agent2["vehicle"])
        if overlap >= min(agent1["vehicle"].area, agent2["vehicle"].area) / 8:
            # Only check for collision if there is an overlap of the safety
            # circle
            p1_a1, p2_a1 = agent1["vehicle"].get_edges()
            p1_a2, p2_a2 = agent2["vehicle"].get_edges()
            collided = False
            for i in range(4):
                if check_intersection_lines(
                    p1_a1, p2_a1, p1_a2[i], p2_a2[i], p2_a1 - p1_a1
                ):
                    collided = True
                    break
            if collided:
                if id1 in rewards:
                    rewards[id1] -= penalty
                    now_done[id1] = True
                if id2 in rewards:
                    rewards[id2] -= penalty
                    now_done[id2] = True
                agent1["done"] = True
                agent2["done"] = True
                self.col_matrix[key] = True

    def check_collision(self, vehicle):
        collided = False
        for a_id in self.get_agent_ids_list():
            if self.agents[a_id] is None:
                continue
            overlap = vehicle.safety_circle_overlap(
                self.agents[a_id]["vehicle"]
            )
            if (
                overlap
                >= min(vehicle.area, self.agents[a_id]["vehicle"].area) / 8
            ):
                p1_a1, p2_a1 = vehicle.get_edges()
                p1_a2, p2_a2 = self.agents[a_id]["vehicle"].get_edges()
                for i in range(4):
                    if check_intersection_lines(
                        p1_a1, p2_a1, p1_a2[i], p2_a2[i], p2_a1 - p1_a1
                    ):
                        collided = True
                        break
            if collided:
                return True
        return False

    def is_done(self):
        all_done = True
        dones = {}
        for id in self.get_agent_ids_list():
            dones[id] = torch.as_tensor(self.agents[id]["done"])
            all_done = all_done and dones[id]
        dones["__all__"] = torch.as_tensor(all_done)
        if self.nsteps >= self.horizon:
            dones["__all__"] = True
        return dones

    def render(self, *args, **kwargs):
        self.world.render(*args, **kwargs)

    def step(
        self, actions, timesteps=10, render=False, tolerance=4.0, **kwargs
    ):
        self.update_env_state()
        states = self.prev_state
        # The actions and states are ideally in global frame
        actions, states, extras = self.transform_state_action(
            actions, states, timesteps
        )

        id_list = self.get_agent_ids_list()
        rewards = {id: 0.0 for id in id_list}

        no_mpc = True
        for ex in extras.values():
            if ex is None:
                no_mpc = False
        intermediates = {}
        if not no_mpc:
            intermediates = self.world.step(states, actions, timesteps)

        for k, val in extras.items():
            if val is not None:
                intermediates[k] = val

        for i in range(0, timesteps):
            no_updates = True
            for a_id in self.get_agent_ids_list():
                if a_id not in intermediates:
                    continue
                if not self.is_agent_done(a_id):
                    state = intermediates[a_id][0][i + 1, :]
                    no_updates = False
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
                    # This is needed only for the MPC code which we are
                    # not using currently
                    # if (
                    #     (state[:2] - actions[a_id][:2]) ** 2
                    # ).sum().sqrt() < tolerance:
                    #     self.world.update_state(
                    #         a_id, state, change_road_association=True,
                    #     )
                    #     continue
                    if i != timesteps - 1:
                        self.world.update_state(a_id, state)
                    else:
                        self.world.update_state(
                            a_id, state, change_road_association=True
                        )
            if no_updates:
                break
            if render:
                self.render(**kwargs)
            intermediate_rewards = self.get_reward()
            for a_id in id_list:
                if a_id not in rewards or a_id not in intermediate_rewards:
                    continue
                rewards[a_id] += intermediate_rewards[a_id]

        self.nsteps += i + 1
        self.world.update_world_state(i + 1)

        self.post_process_rewards(rewards, None)

        return (
            self.get_state(),
            rewards,
            self.is_done(),
            {"timeout": self.horizon < self.nsteps},
        )
