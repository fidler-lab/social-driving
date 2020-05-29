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
from sdriving.trafficsim.dynamics import FixedTrackAccelerationModel
from sdriving.trafficsim.utils import angle_normalize
from sdriving.trafficsim.vehicle import Vehicle
from sdriving.trafficsim.world import World


class RoadIntersectionControlAccelerationEnv(RoadIntersectionControlEnv):
    def __init__(
        self,
        *args,
        fast_model: bool = False,
        has_turns: bool = False,
        **kwargs,
    ):
        self.fast_model = fast_model
        self.has_turns = has_turns
        super().__init__(*args, **kwargs)

    def generate_world_without_agents(self):
        self.length = (torch.rand(1) * 30.0 + 50.0).item()
        self.width = (torch.rand(1) * 20.0 + 10.0).item()
        time_green = int((torch.rand(1) / 2 + 1) * self.time_green)
        if self.has_turns:
            gen = generate_intersection_world_12signals
            ordering = random.choice(range(8))
        else:
            gen = generate_intersection_world_4signals
            ordering = random.choice(range(2))
        return gen(
            length=self.length,
            road_width=self.width,
            name="traffic_signal_world",
            time_green=time_green,
            ordering=ordering,
        )

    def configure_action_list(self):
        if not self.fast_model:
            self.actions_list = [
                torch.as_tensor([[ac]]) for ac in np.arange(-1.5, 1.75, 0.25)
            ]
            self.max_accln = 1.5
        else:
            self.actions_list = [
                torch.as_tensor([[ac]]) for ac in np.arange(-3.0, 3.01, 0.25)
            ]
            self.max_accln = 3.0

    def reset(self):
        self.lane_side = 1.0  # np.random.choice([-1.0, 1.0])
        return super().reset()

    @staticmethod
    def _get_ranges_road(rd):
        coord = rd.coordinates
        x_range = (
            torch.min(coord[:, 0]).item(),
            torch.max(coord[:, 0]).item(),
        )
        y_range = (
            torch.min(coord[:, 1]).item(),
            torch.max(coord[:, 1]).item(),
        )
        return (x_range, y_range)

    def _get_track(self, world, width, rd1, rd2, pos, t1, t2):
        rwd_2 = width / 2

        track = dict()
        # Central Part for turning / going straight
        if (int(rd1[-1]) + 2) % 4 == int(rd2[-1]):
            # Need to go straight
            track[((-rwd_2, rwd_2), (-rwd_2, rwd_2))] = {
                "turn": False,
                "theta": t1,
            }
        else:
            clockwise = (int(rd2[-1]) - int(rd1[-1]) + 4) % 4 == 1
            # 3 for anticlockwise
            rd = rd1 if clockwise else rd2
            center = world.road_network.roads[rd].coordinates[1].clone()
            r = pos[(int(rd1[-1]) + 1) % 2].clone()
            if rd1[-1] in ("0", "3"):
                r *= -1 if clockwise else 1
            else:
                r *= 1 if clockwise else -1
            r += width / 2
            track[((-rwd_2, rwd_2), (-rwd_2, rwd_2))] = {
                "turn": True,
                "center": center,
                "radius": r,
                "clockwise": clockwise,
            }

        # Inside the starting road
        track[self._get_ranges_road(world.road_network.roads[rd1])] = {
            "turn": False,
            "theta": t1,
        }

        # Inside the ending road
        track[self._get_ranges_road(world.road_network.roads[rd2])] = {
            "turn": False,
            "theta": t2,
        }

        return track

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
                    side = self.lane_side * (
                        1 if srd[-1] in ("1", "2") else -1
                    )
                    spos[(int(srd[-1]) + 1) % 2] = (
                        side * (torch.rand(1) * 0.15 + 0.15) * self.width
                    )
            else:
                spos = sroad.offset.clone()
        else:
            if sample:
                s_pos = sroad.sample(x_bound=0.6, y_bound=0.6)[0]
                if hasattr(self, "lane_side"):
                    side = self.lane_side * (
                        1 if srd[-1] in ("1", "2") else -1
                    )
                    s_pos[(int(srd[-1]) + 1) % 2] = (
                        side * (torch.rand(1) * 0.15 + 0.15) * self.width
                    )
            else:
                s_pos = sroad.offset.clone()
            spos[int(srd[-1]) % 2] = s_pos[int(srd[-1]) % 2]

        track = self._get_track(
            self.world,
            self.width,
            srd,
            erd,
            spos,
            orientation,
            dest_orientation,
        )

        if place:
            self.add_vehicle(
                a_id,
                srd,
                spos,
                torch.as_tensor(20.0 if self.fast_model else 8.0),
                orientation,
                end_pos,
                dest_orientation,
                dynamics_model=FixedTrackAccelerationModel,
                dynamics_kwargs={"track": track},
            )
        else:
            return (
                a_id,
                srd,
                spos,
                torch.as_tensor(20.0 if self.fast_model else 8.0),
                orientation,
                end_pos,
                dest_orientation,
                FixedTrackAccelerationModel,
                {"track": track},
            )

    def setup_nagents_1(self):
        if not self.has_turns:
            super().setup_nagents_1()
            return
        self.setup_nagents(1)

    def setup_nagents_2(self, **kwargs):
        if not self.has_turns:
            super().setup_nagents_2()
            return
        self.setup_nagents(2)

    def end_road_sampler(self, n: int):
        if not self.has_turns:
            return super().end_road_sampler(n)
        return np.random.choice(list(set(range(4)) - set([n])))

    def post_process_rewards(self, rewards, now_dones):
        # Encourage the agents to make smoother transitions
        for a_id in self.get_agent_ids_list():
            if a_id not in rewards:
                continue
            rew = rewards[a_id]
            if a_id not in self.prev_actions or a_id not in self.curr_actions:
                # Agents are removed in case of Continuous Flow Environments
                continue
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
            penalty = diff[0] / (4 * self.max_accln * self.horizon)
            rewards[a_id] = rew - penalty


class RoadIntersectionContinuousAccelerationEnv(
    RoadIntersectionControlAccelerationEnv
):
    def get_action_space(self):
        self.max_accln = 2.5
        return Box(low=-2.5, high=2.5, shape=(1,))

    def transform_state_action_single_agent(
        self, a_id: str, action: torch.Tensor, state, timesteps: int
    ):
        agent = self.agents[a_id]["vehicle"]

        x, y = agent.position
        v = agent.speed 
        t = agent.orientation 

        start_state = torch.as_tensor([x, y, v, t])
        dynamics = self.agents[a_id]["dynamics"]
        nominal_states = [start_state.unsqueeze(0)]
        nominal_actions = [action]

        action.unsqueeze_(0)
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
                    side = self.lane_side * (
                        1 if srd[-1] in ("1", "2") else -1
                    )
                    spos[(int(srd[-1]) + 1) % 2] = side * self.width / 4
            else:
                spos = sroad.offset.clone()
        else:
            if sample:
                s_pos = sroad.sample(x_bound=0.6, y_bound=0.6)[0]
                if hasattr(self, "lane_side"):
                    side = self.lane_side * (
                        1 if srd[-1] in ("1", "2") else -1
                    )
                    spos[(int(srd[-1]) + 1) % 2] = side * self.width / 4
            else:
                s_pos = sroad.offset.clone()
            spos[int(srd[-1]) % 2] = s_pos[int(srd[-1]) % 2]

        track = self._get_track(
            self.world,
            self.width,
            srd,
            erd,
            spos,
            orientation,
            dest_orientation,
        )

        if place:
            self.add_vehicle(
                a_id,
                srd,
                spos,
                torch.as_tensor(20.0 if self.fast_model else 8.0),
                orientation,
                end_pos,
                dest_orientation,
                dynamics_model=FixedTrackAccelerationModel,
                dynamics_kwargs={"track": track},
            )
        else:
