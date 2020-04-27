import itertools
import math
import random
from collections import deque

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple

from sdriving.envs.base_env import BaseEnv
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


class RoadIntersectionEnv(BaseEnv):
    def __init__(
        self,
        hybrid_controller: str,
        npoints: int = 100,
        horizon: int = 200,
        tolerance: float = 0.0,
        object_collision_penalty: float = 1.0,
        agents_collision_penalty: float = 1.0,
        goal_reach_bonus: float = 1.0,
        simple: bool = True,
        history_len: int = 5,
        time_green: int = 75,
        nagents: int = 4,
        device=torch.device("cpu"),
        lidar_range: float = 50.0,
        lidar_noise: float = 0.0,
        mode: int = 1,
        has_lane_distance: bool = False,
        balance_cars: bool = False,
    ):
        self.npoints = npoints
        self.goal_reach_bonus = goal_reach_bonus
        self.simple = simple
        self.history_len = history_len
        self.time_green = time_green
        self.device = device
        self.hybrid_controller = hybrid_controller
        self.has_lane_distance = has_lane_distance
        world = self.generate_world_without_agents()
        super().__init__(
            world,
            nagents,
            horizon,
            tolerance,
            object_collision_penalty,
            agents_collision_penalty,
            True,
        )
        # We need to store the history to stack them. 1 queue stores the
        # observation and the other stores the lidar data
        self.queue1 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.queue2 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.lidar_range = lidar_range
        self.lidar_noise = lidar_noise
        self.mode = mode
        self.prev_actions = {
            a_id: torch.zeros(2) for a_id in self.get_agent_ids_list()
        }
        self.curr_actions = {a_id: None for a_id in self.get_agent_ids_list()}
        self.balance_cars = balance_cars

    def get_next_two_goals(
        self, a_id=None, prev_point=None, intermediate_nodes=None
    ):
        if a_id is not None:
            prev_point = self.agents[a_id]["prev_point"]
            intermediate_nodes = self.agents[a_id]["intermediate_nodes"]
        if prev_point >= len(intermediate_nodes):
            return -1, -1
        elif prev_point == len(intermediate_nodes) - 1:
            return intermediate_nodes[-1], -1
        return (
            intermediate_nodes[prev_point],
            intermediate_nodes[prev_point + 1],
        )

    def generate_world_without_agents(self):
        self.length = (torch.rand(1) * 30.0 + 40.0).item()
        self.width = (torch.rand(1) * 20.0 + 10.0).item()
        time_green = int((torch.rand(1) / 2 + 1) * self.time_green)
        return generate_intersection_world_4signals(
            length=self.length,
            road_width=self.width,
            name="traffic_signal_world",
            time_green=time_green,
            ordering=random.choice([0, 1]),
        )

    def configure_action_list(self):
        self.actions_list = [
            torch.as_tensor(ac)
            for ac in itertools.product([-10.0, 10.0], [-10.0, 10.0])
        ]
        self.actions_list += [
            torch.as_tensor(ac)
            for ac in [(0.0, -10.0), (0.0, 10.0), (-10.0, 0.0), (10.0, 0.0)]
        ]

    def get_action_space(self):
        return Discrete(len(self.actions_list) + 1)

    def get_observation_space(self):
        if self.has_lane_distance:
            return Tuple(
                [
                    Box(
                        # Distance from Lane, Traffic Signal, Velocity Diff,
                        # Optimal Heading, Inverse Distance
                        low=np.array(
                            [-np.inf, 0.0, -1.0, -1.0, 0.0] * self.history_len
                        ),
                        high=np.array(
                            [np.inf, 1.0, 1.0, 1.0, np.inf] * self.history_len
                        ),
                    ),
                    Box(0.0, np.inf, shape=(self.npoints * self.history_len,)),
                ]
            )
        else:
            return Tuple(
                [
                    Box(
                        # Signal, Velocity Diff, Optimal Heading, Inverse Distance
                        low=np.array(
                            [0.0, -1.0, -1.0, 0.0] * self.history_len
                        ),
                        high=np.array(
                            [1.0, 1.0, 1.0, np.inf] * self.history_len
                        ),
                    ),
                    Box(0.0, np.inf, shape=(self.npoints * self.history_len,)),
                ]
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
        dynamics_model=VehicleDynamics,
        dynamics_kwargs={},
    ):
        vehicle = Vehicle(
            pos,
            orientation,
            destination=dest,
            dest_orientation=dest_orientation,
            name=a_id,
            max_lidar_range=self.lidar_range,
        )
        dynamics = dynamics_model(
            dim=[vehicle.dimensions[0]],
            v_lim=[v_lim.item()],
            **dynamics_kwargs,
        )

        if hasattr(self, "hybrid_controller"):
            controller = HybridController(
                self.world, self.hybrid_controller, npoints=self.npoints
            )
        else:
            controller = None

        # Register the vehicle with the world
        self.world.add_vehicle(vehicle, rname, v_lim=v_lim.item())

        # Store local references to the agent as well
        intermediate_nodes = self.world.shortest_path_trajectory(
            vehicle.position, vehicle.destination, vehicle.name
        )

        self.agents[a_id] = {
            "vehicle": vehicle,
            "done": False,
            "v_lim": v_lim.item(),
            "dynamics": dynamics,
            "controller": controller,
            "original_destination": dest,
            "dest_orientation": dest_orientation,
            "original_distance": vehicle.distance_from_destination(),
            "intermediate_nodes": intermediate_nodes,
            "intermediate_goals": [
                *[
                    torch.as_tensor([*i, vehicle.speed, vehicle.orientation])
                    for i in self.world.road_network.all_nodes[
                        intermediate_nodes
                    ]
                ],
                torch.as_tensor(
                    [*vehicle.destination, 0.0, vehicle.dest_orientation,]
                ),
            ],
            "prev_point": 0,
            "goal_reach_bonus": False,
        }

        int_goals = self.agents[a_id]["intermediate_goals"]
        distances = np.array(
            [
                ((pt1[:2] - pt2[:2]) ** 2).sum().sqrt().item()
                for pt1, pt2 in zip(int_goals[:-1], int_goals[1:])
            ]
        )
        distances = np.flip(np.cumsum(np.flip(distances, 0)), 0)
        self.agents[a_id]["distances"] = distances
        self.agents[a_id]["original_distance"] = (
            distances[0]
            + ((vehicle.position - int_goals[0][:2]) ** 2).sum().sqrt()
        )

    def get_state_single_agent(self, a_id):
        agent = self.agents[a_id]["vehicle"]
        v_lim = self.agents[a_id]["v_lim"]

        if self.agents[a_id]["prev_point"] < len(
            self.agents[a_id]["intermediate_goals"]
        ):
            xg, yg, vg, _ = self.agents[a_id]["intermediate_goals"][
                self.agents[a_id]["prev_point"]
            ]
            dest = torch.as_tensor([xg, yg])
        else:
            dest = agent.destination
            vg = 0.0

        inv_dist = 1 / agent.distance_from_point(dest)
        pt1, pt2 = self.get_next_two_goals(a_id)
        if self.has_lane_distance:
            obs = [
                self.world.get_distance_from_road_axis(
                    a_id, pt1, self.agents[a_id]["original_destination"]
                ),
                self.world.get_traffic_signal(
                    pt1, pt2, agent.position, agent.vision_range
                ),
                (vg - agent.speed) / (2 * v_lim),
                agent.optimal_heading_to_point(dest) / math.pi,
                inv_dist if torch.isfinite(inv_dist) else 0.0,
            ]
        else:
            obs = [
                self.world.get_traffic_signal(
                    pt1, pt2, agent.position, agent.vision_range
                ),
                (vg - agent.speed) / (2 * v_lim),
                agent.optimal_heading_to_point(dest) / math.pi,
                inv_dist if torch.isfinite(inv_dist) else 0.0,
            ]
        cur_state = [
            torch.as_tensor(obs),
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

    def transform_state_action_single_agent(
        self, a_id, action, state, timesteps
    ):
        agent = self.agents[a_id]["vehicle"]

        x, y = agent.position
        v = agent.speed
        t = agent.orientation

        if action > 0:
            dx, dy = self.actions_list[action - 1]
            ct = torch.cos(t)
            st = torch.sin(t)
            delx = dx * ct - dy * st
            dely = dx * st + dy * ct
            goal_state = torch.as_tensor([x + delx, y + dely, v, t,])

            na, ns, ex = self.agents[a_id]["controller"](
                torch.as_tensor([x, y, v, t]),
                goal_state,
                agent,
                self.agents[a_id]["dynamics"],
                {"v_lim": self.agents[a_id]["v_lim"]},
            )
        else:
            start_state = torch.as_tensor([x, y, v, t])
            action = torch.zeros((1, 2))
            dynamics = self.agents[a_id]["dynamics"]
            nominal_states = [start_state.unsqueeze(0)]
            nominal_actions = [action]
            for _ in range(timesteps):
                start_state = nominal_states[-1]
                new_state = dynamics(start_state, action)
                new_state[0, 2] /= 1.05
                nominal_states.append(new_state.cpu())
                nominal_actions.append(action)
            nominal_states, nominal_actions = (
                torch.cat(nominal_states),
                torch.cat(nominal_actions),
            )
            na = torch.zeros(4).to(self.agents[a_id]["controller"].device)
            ns = torch.zeros(4).to(self.agents[a_id]["controller"].device)
            ex = (nominal_states, nominal_actions)

        return na, ns, ex

    def handle_goal_tolerance(self, agent):
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
        if agent["prev_point"] < len(agent["intermediate_goals"]):
            idx = agent["prev_point"]
            dest = agent["intermediate_goals"][idx][:2]
            if idx < len(agent["distances"]):
                add_dist = agent["distances"][idx]
            else:
                add_dist = 0.0
        else:
            idx = None
            dest = agent["vehicle"].destination
            return 0.0
        dist = agent["vehicle"].distance_from_point(dest)
        pt1, _ = self.get_next_two_goals(
            prev_point=agent["prev_point"],
            intermediate_nodes=agent["intermediate_nodes"],
        )
        ld = self.world.get_distance_from_road_axis(
            agent["vehicle"].name, pt1, agent["original_destination"]
        )
        dist = (dist ** 2 - ld ** 2).sqrt()

        return (dist + add_dist) / (
            (agent["original_distance"] + 1e-12) * self.horizon
        )

    def add_vehicle_path(
        self,
        a_id: str,
        srd: int,
        erd: int,
        sample: bool,
        spos=None,
        place: bool = True,
    ):
        srd = f"traffic_signal_world_{srd}"
        sroad = self.world.road_network.roads[srd]
        erd = f"traffic_signal_world_{erd}"
        eroad = self.world.road_network.roads[erd]

        orientation = angle_normalize(sroad.orientation + math.pi).item()
        dest_orientation = eroad.orientation.item()

        end_pos = torch.zeros(2)
        if erd[-1] in ("1", "3"):
            end_pos[1] = (
                (self.length / 1.5 + self.width)
                / 2
                * (1 if erd[-1] == "1" else -1)
            )
        else:
            end_pos[0] = (
                (self.length / 1.5 + self.width)
                / 2
                * (1 if erd[-1] == "0" else -1)
            )

        if spos is None:
            if sample:
                spos = sroad.sample(x_bound=0.6, y_bound=0.6)[0]
                if hasattr(self, "lane_side"):
                    side = self.lane_side * (1 if srd[-1] in ("1", "2") else -1)
                    spos[(int(srd[-1]) + 1) % 2] = (
                        side * (torch.rand(1) * 0.15 + 0.15) * self.width
                    )
            else:
                spos = sroad.offset.clone()

        if place:
            self.add_vehicle(
                a_id,
                srd,
                spos,
                torch.as_tensor(8.0),
                orientation,
                end_pos,
                dest_orientation,
                VehicleDynamics,
                {},
            )
        else:
            return (
                a_id,
                srd,
                spos,
                torch.as_tensor(8.0),
                orientation,
                end_pos,
                dest_orientation,
                VehicleDynamics,
                {},
            )

    def setup_nagents_1(self):
        # Start at the road "traffic_signal_0" as the state space is
        # invariant to rotation and start position
        erd = np.random.choice([1, 2, 3])
        a_id = self.get_agent_ids_list()[0]

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
            self.add_vehicle_path(self.get_agent_ids_list()[0], 0, 2, sample)

            # Choose which road the next car is placed
            srd = np.random.choice([1, 3])
            self.add_vehicle_path(
                self.get_agent_ids_list()[1], srd, (srd + 2) % 4, sample
            )
        elif mode == 2:
            # Parallel cars. Learn only lane following
            self.add_vehicle_path(self.get_agent_ids_list()[0], 0, 2, sample)
            self.add_vehicle_path(self.get_agent_ids_list()[1], 2, 0, sample)
        elif mode == 3:
            # Sample uniformly between modes 1 and 2
            self.setup_nagents_2(
                bypass_mode=np.random.choice([1, 2]), sample=sample
            )

    @staticmethod
    def end_road_sampler(n: int):
        return (n + 2) % 4

    def setup_nagents(self, n: int):
        # Try to place the agent without any overlap
        sample = False if self.mode == 1 else True
        if self.mode not in [1, 2]:
            raise NotImplementedError

        placed = 0
        srd = np.random.choice([0, 1, 2, 3])
        while placed < n:
            if self.balance_cars:
                srd = (srd + 1) % 4
            else:
                srd = np.random.choice([0, 1, 2, 3])
            erd = self.end_road_sampler(srd)

            free = False
            while not free:
                (
                    a_id,
                    rd,
                    spos,
                    vlim,
                    orientation,
                    end_pos,
                    dest_orientation,
                    dynamics_model,
                    dynamics_kwargs,
                ) = self.add_vehicle_path(
                    self.get_agent_ids_list()[placed],
                    srd,
                    erd,
                    sample,
                    place=False,
                )
                vehicle = Vehicle(
                    spos,
                    orientation,
                    destination=end_pos,
                    dest_orientation=dest_orientation,
                    name=a_id,
                    max_lidar_range=self.lidar_range,
                )
                free = not self.check_collision(vehicle)
            self.add_vehicle(
                a_id,
                rd,
                spos,
                vlim,
                orientation,
                end_pos,
                dest_orientation,
                dynamics_model,
                dynamics_kwargs,
            )
            placed += 1

    def reset(self):
        # Keep the environment fixed for now
        self.world = self.generate_world_without_agents()

        self.queue1 = {
            a_id: deque(maxlen=self.history_len)
            for a_id in self.get_agent_ids_list()
        }
        self.queue2 = {
            a_id: deque(maxlen=self.history_len)
            for a_id in self.get_agent_ids_list()
        }

        if self.nagents == 1:
            self.setup_nagents_1()
        elif self.nagents == 2:
            self.setup_nagents_2()
        else:
            self.setup_nagents(self.nagents)

        return super().reset()


class RoadIntersectionControlEnv(RoadIntersectionEnv):
    def __init__(
        self,
        npoints: int = 100,
        horizon: int = 200,
        tolerance: float = 0.0,
        object_collision_penalty: float = 1.0,
        agents_collision_penalty: float = 1.0,
        goal_reach_bonus: float = 1.0,
        simple: bool = True,
        history_len: int = 5,
        time_green: int = 75,
        nagents: int = 4,
        device=torch.device("cpu"),
        lidar_range: float = 50.0,
        lidar_noise: float = 0.0,
        mode: int = 1,
        has_lane_distance: bool = False,
        balance_cars: bool = False,
    ):
        self.npoints = npoints
        self.goal_reach_bonus = goal_reach_bonus
        self.simple = simple
        self.history_len = history_len
        self.time_green = time_green
        self.device = device
        self.has_lane_distance = has_lane_distance
        world = self.generate_world_without_agents()
        super(RoadIntersectionEnv, self).__init__(
            world,
            nagents,
            horizon,
            tolerance,
            object_collision_penalty,
            agents_collision_penalty,
            True,
        )
        # We need to store the history to stack them. 1 queue stores the
        # observation and the other stores the lidar data
        self.queue1 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.queue2 = {a_id: None for a_id in self.get_agent_ids_list()}
        self.lidar_range = lidar_range
        self.lidar_noise = lidar_noise
        self.mode = mode
        self.prev_actions = {
            a_id: torch.zeros(2) for a_id in self.get_agent_ids_list()
        }
        self.curr_actions = {a_id: None for a_id in self.get_agent_ids_list()}
        self.balance_cars = balance_cars

    def configure_action_list(self):
        self.actions_list = [
            torch.as_tensor(ac).unsqueeze(0)
            for ac in itertools.product(
                [-0.1, -0.05, 0.0, 0.05, 0.1], [-1.5, -0.75, 0.0, 0.75, 1.5],
            )
        ]

    def get_action_space(self):
        return Discrete(len(self.actions_list))

    def post_process_rewards(self, rewards, now_dones):
        # Encourage the agents to make smoother transitions
        for a_id, rew in rewards.items():
            pac = self.prev_actions[a_id]
            if now_dones is None:
                # The penalty should be lasting for all the
                # timesteps the action was taken. None is passed
                # when the intermediate timesteps have been
                # processed, so swap the actions here
                self.curr_actions[a_id] = pac
                return
            cac = self.curr_actions[a_id]
            diff = torch.abs(pac - cac)
            penalty = (diff[0] / 0.2 + diff[1] / 3.0) / (2 * self.horizon)
            rewards[a_id] = rew - penalty

    def transform_state_action_single_agent(
        self, a_id, action, state, timesteps
    ):
        agent = self.agents[a_id]["vehicle"]

        x, y = agent.position
        v = agent.speed
        t = agent.orientation

        action = self.actions_list[action]
        start_state = torch.as_tensor([x, y, v, t])
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
        na = torch.zeros(4).to(self.device)
        ns = torch.zeros(4).to(self.device)
        ex = (nominal_states, nominal_actions)

        self.curr_actions[a_id] = action[0]

        return na, ns, ex


class RoadIntersectionControlImitateEnv(RoadIntersectionControlEnv):
    def __init__(self, base_model: str, *args, lam: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        device = kwargs.get("device", torch.device("cpu"))
        ckpt = torch.load(base_model, map_location="cpu")
        if "model" not in ckpt or ckpt["model"] == "centralized_critic":
            from sdriving.agents.ppo_cent.model import ActorCritic
        ac = ActorCritic(**ckpt["ac_kwargs"])
        ac.v = None
        ac.pi.load_state_dict(ckpt["actor"])
        ac = ac.to(device)
        self.base_model = ac
        # We need to store the history to stack them. 1 queue stores the
        # observation and the other stores the lidar data
        self.queue1_bm = {a_id: None for a_id in self.get_agent_ids_list()}
        self.queue2_bm = {a_id: None for a_id in self.get_agent_ids_list()}
        self.recompute_base_model_actions = True
        self.base_model_actions = None
        self.lam = lam

    def post_process_rewards(self, rewards, now_dones):
        super().post_process_rewards(rewards, now_dones)
        if now_dones is None:
            self.recompute_base_model_actions = True
            return

        if self.recompute_base_model_actions:
            base_model_state = self._get_state_base_model()
            self.base_model_actions = dict()
            for key, obs in base_model_state.items():
                base_model_state[key] = [t.to(self.device) for t in obs]
                # Get the deterministic actions from the base model
                self.base_model_actions[key] = self.actions_list[
                    self.base_model.act(base_model_state[key], True).item()
                ][0]

        # Try to imitate the behavior of an agent as if it is driving
        # in an empty environment
        for a_id, rew in rewards.items():
            bac = self.base_model_actions[a_id]
            cac = self.curr_actions[a_id]
            diff = torch.abs(bac - cac)
            # TODO: Tune the weight on this penalty.
            penalty = (
                self.lam * (diff[0] / 0.2 + diff[1] / 3.0) / (2 * self.horizon)
            )
            rewards[a_id] = rew - penalty

    def reset(self):
        self.queue1_bm = {
            a_id: deque(maxlen=self.history_len)
            for a_id in self.get_agent_ids_list()
        }
        self.queue2_bm = {
            a_id: deque(maxlen=self.history_len)
            for a_id in self.get_agent_ids_list()
        }
        return super().reset()

    def _get_state_base_model(self):
        return {
            a_id: self._get_state_single_agent_base_model(a_id)
            for a_id in self.get_agent_ids_list()
        }

    def _get_state_single_agent_base_model(self, a_id):
        agent = self.agents[a_id]["vehicle"]
        v_lim = self.agents[a_id]["v_lim"]

        if self.agents[a_id]["prev_point"] < len(
            self.agents[a_id]["intermediate_goals"]
        ):
            xg, yg, vg, _ = self.agents[a_id]["intermediate_goals"][
                self.agents[a_id]["prev_point"]
            ]
            dest = torch.as_tensor([xg, yg])
        else:
            dest = agent.destination
            vg = 0.0

        inv_dist = 1 / agent.distance_from_point(dest)
        pt1, pt2 = self.get_next_two_goals(a_id)
        if self.has_lane_distance:
            obs = [
                self.world.get_distance_from_road_axis(
                    a_id, pt1, self.agents[a_id]["original_destination"]
                ),
                self.world.get_traffic_signal(
                    pt1, pt2, agent.position, agent.vision_range
                ),
                (vg - agent.speed) / (2 * v_lim),
                agent.optimal_heading_to_point(dest) / math.pi,
                inv_dist if torch.isfinite(inv_dist) else 0.0,
            ]
        else:
            obs = [
                self.world.get_traffic_signal(
                    pt1, pt2, agent.position, agent.vision_range
                ),
                (vg - agent.speed) / (2 * v_lim),
                agent.optimal_heading_to_point(dest) / math.pi,
                inv_dist if torch.isfinite(inv_dist) else 0.0,
            ]
        cur_state = [
            torch.as_tensor(obs),
            1
            / self.world.get_lidar_data(agent.name, self.npoints, cars=False),
        ]

        if self.lidar_noise != 0.0:
            cur_state[1] *= torch.rand(self.npoints) > self.lidar_noise

        while len(self.queue1_bm[a_id]) <= self.history_len - 1:
            self.queue1_bm[a_id].append(cur_state[0])
            self.queue2_bm[a_id].append(cur_state[1])
        self.queue1_bm[a_id].append(cur_state[0])
        self.queue2_bm[a_id].append(cur_state[1])

        return (
            torch.cat(list(self.queue1_bm[a_id])),
            torch.cat(list(self.queue2_bm[a_id])),
        )
