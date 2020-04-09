from itertools import combinations

import torch
from tqdm import tqdm

from sdriving.trafficsim import World
from sdriving.trafficsim.dynamics import (
    BicycleKinematicsModel as VehicleDynamics,
)
from sdriving.trafficsim.utils import check_intersection_lines
from sdriving.trafficsim.vehicle import Vehicle


class WorldSimulator:
    def __init__(self, world: World, device=torch.device("cpu")):
        self.world = world

        # Dict[agent_id] -> List[(state, action)]
        self.state_action_map = {}

        # Dict[agent_id] -> {"vehicle", "dynamics", "controller"}
        self.agents = {}

        self.device = device

    def reset(self):
        self.world.reset()
        self.state_action_map = {}
        self.agents = {}

    def add_vehicle(
        self,
        name: str,
        rname: str,
        vehicle: Vehicle,
        dynamics: VehicleDynamics,
        controller,
    ):
        self.agents[name] = {
            "vehicle": vehicle,
            "dynamics": dynamics,
            "controller": controller,
        }

        self.world.add_vehicle(vehicle, rname, v_lim=dynamics.v_lim)

        controller.initialize(vehicle)

    def compile(self):
        self.world.compile()
        self.world.to(self.device)

    def intervehicle_collision(self, i1, i2):
        id_list = list(self.agents.keys())
        id1 = id_list[i1]
        id2 = id_list[i2]
        agent1 = self.agents[id1]
        agent2 = self.agents[id2]
        overlap = agent1["vehicle"].safety_circle_overlap(agent2["vehicle"])
        collided = False
        if overlap >= min(agent1["vehicle"].area, agent2["vehicle"].area) / 8:
            p1_a1, p2_a1 = agent1["vehicle"].get_edges()
            p1_a2, p2_a2 = agent2["vehicle"].get_edges()
            for i in range(4):
                if check_intersection_lines(
                    p1_a1, p2_a1, p1_a2[i], p2_a2[i], p2_a1 - p1_a1
                ):
                    collided = True
                    break
        return collided

    def simulate(
        self,
        total_timesteps,
        store_actions=False,
        render_level=2,
        render_path=None,
    ):
        tstep = 0
        all_done = False
        dones = {a_id: False for a_id in self.agents}

        goal_states = {
            a_id: torch.as_tensor(
                [
                    *fields["vehicle"].destination,
                    0.0,
                    fields["vehicle"].dest_orientation,
                ]
            ).to(self.device)
            for a_id, fields in self.agents.items()
        }

        nagents = len(goal_states.keys())

        vehicle_collision_matrix = torch.zeros(nagents, nagents).bool()

        for a_id in self.agents:
            self.state_action_map[a_id] = []

        pbar = tqdm(total=total_timesteps)
        while tstep < total_timesteps and not all_done:
            start_states = {
                a_id: torch.as_tensor(
                    [
                        *fields["vehicle"].position,
                        fields["vehicle"].speed,
                        fields["vehicle"].orientation,
                    ]
                ).to(self.device)
                for a_id, fields in self.agents.items()
            }

            local_goals, local_starts, local_extras = {}, {}, {}
            no_mpc = True

            # Get the agent predictions for waypoints
            for a_id in start_states.keys():
                (
                    local_goals[a_id],
                    local_starts[a_id],
                    local_extras[a_id],
                    act,
                ) = self.agents[a_id]["controller"](
                    start_states[a_id],
                    goal_states[a_id],
                    self.agents[a_id]["vehicle"],
                    10,
                    self.agents[a_id]["dynamics"],
                    return_action=True,
                )
                if store_actions:
                    self.state_action_map[a_id].append(
                        (
                            self.agents[a_id]["controller"].get_agent_state(
                                start_states[a_id],
                                goal_states[a_id],
                                self.agents[a_id]["vehicle"],
                                self.agents[a_id]["dynamics"],
                                modify_buffer=False,
                            ),
                            act,
                        )
                    )
                if local_extras[a_id] is None:
                    no_mpc = False

            # Get the final positions after running the MPC
            states = {}
            if not no_mpc:
                states = self.world.step(local_starts, local_goals, 10)

            for a_id, ex in local_extras.items():
                if ex is not None:
                    states[a_id] = ex

            # Modify the world and controller states
            for a_id in start_states.keys():
                self.agents[a_id]["controller"].postprocess_states(
                    states[a_id][0], a_id
                )

            for i in range(10):
                for a_id, state in states.items():
                    if not dones[a_id]:
                        if i == state[0].size(0) - 1:
                            self.world.update_state(
                                a_id, state[0][i], change_road_association=True
                            )
                        else:
                            self.world.update_state(a_id, state[0][i])
                if render_level == 2:
                    self.world.render()

            if render_level == 1:
                self.world.render()

            # Check collision
            for a_id in self.agents.keys():
                if not dones[a_id]:
                    dones[a_id] = self.world.check_collision(a_id)
            for i1, i2 in combinations(range(nagents), 2):
                if not (
                    vehicle_collision_matrix[i1][i2]
                    or vehicle_collision_matrix[i2][i1]
                ):
                    if self.intervehicle_collision(i1, i2):
                        vehicle_collision_matrix[i1][i2] = True
                        vehicle_collision_matrix[i2][i1] = True
            tstep += 10
            self.world.update_world_state(10)
        pbar.update(10)
        pbar.close()

        if render_level >= 1:
            self.world.render(path=render_path)
