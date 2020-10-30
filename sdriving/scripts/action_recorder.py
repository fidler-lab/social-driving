import argparse
import json
import math

import gym
import pandas as pd
import torch

from sdriving.scripts.rollout import RolloutSimulator
from sdriving.tsim import angle_normalize

DEFAULT_RECORD_LIST = [
    "Velocity",
    "Acceleration",
    "Time Step",
    "Heading",
    "Episode",
    "Agent ID",
    "Minimum Distance to Car",
    "Relative Velocity",
    "Position",
]

ENV2RECORD = dict()

EXTRAS = dict(
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment=[
        "Traffic Signal",
        "Steering Angle",
    ],
    MultiAgentRoadIntersectionBicycleKinematicsDiscreteEnvironment=[
        "Traffic Signal",
        "Steering Angle",
    ],
    MultiAgentRoadIntersectionFixedTrackEnvironment=["Traffic Signal"],
    MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment=["Traffic Signal"],
    MultiAgentRoadIntersectionFixedTrackDiscreteCommunicationEnvironment=[
        "Traffic Signal",
        "Communication (Recv)",
        "Communication (Send)",
    ],
    MultiAgentIntersectionSplineAccelerationDiscreteEnvironment=[
        "Traffic Signal"
    ],
    MultiAgentIntersectionSplineAccelerationDiscreteV2Environment=[
        "Traffic Signal"
    ],
    MultiAgentNuscenesIntersectionDrivingEnvironment=["Traffic Signal"],
    MultiAgentNuscenesIntersectionDrivingDiscreteEnvironment=[
        "Traffic Signal"
    ],
    MultiAgentNuscenesIntersectionDrivingCommunicationDiscreteEnvironment=[
        "Traffic Signal",
        "Communication (Recv)",
        "Communication (Send)",
    ],
    MultiAgentNuscenesIntersectionBicycleKinematicsEnvironment=[
        "Traffic Signal",
        "Steering Angle",
    ],
    MultiAgentNuscenesIntersectionBicycleKinematicsDiscreteEnvironment=[
        "Traffic Signal",
        "Steering Angle",
    ],
    MultiAgentHighwayBicycleKinematicsModel=["Acceleration Rating"],
    MultiAgentHighwayBicycleKinematicsDiscreteModel=["Acceleration Rating"],
    MultiAgentHighwaySplineAccelerationDiscreteModel=["Acceleration Rating"],
    MultiAgentHighwayPedestriansFixedTrackDiscreteModel=[
        "Distance to Crosswalk",
        "Distance to Nearest Pedestrian",
    ],
    MultiAgentHighwayPedestriansSplineAccelerationDiscreteModel=[
        "Distance to Crosswalk",
        "Distance to Nearest Pedestrian",
        "Acceleration Rating",
    ],
)

for k, v in EXTRAS.items():
    ENV2RECORD[k] = DEFAULT_RECORD_LIST + v


class RolloutSimulatorActionRecorder(RolloutSimulator):
    def __init__(self, fname: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fname = self.save_dir / fname

        # Parse the record items and check which quantities to store
        # Only negate the extra quatities
        self.record_items = ENV2RECORD[self.env_name]

        self.record_steering = "Steering Angle" in self.record_items
        self.record_global_position = "Position" in self.record_items
        self.record_accln_rating = "Acceleration Rating" in self.record_items
        self.record_traffic_signal = "Traffic Signal" in self.record_items
        self.record_recv_communication = (
            "Communication (Recv)" in self.record_items
        )
        self.record_send_communication = (
            "Communication (Send)" in self.record_items
        )
        self.record_min_distance_to_car = (
            "Minimum Distance to Car" in self.record_items
        )
        self.record_pedestrian_distance = (
            "Distance to Nearest Pedestrian" in self.record_items
        )
        self.record_distance_to_crosswalk = (
            "Distance to Crosswalk" in self.record_items
        )

        self.record = {r: [] for r in self.record_items}
        self.episode_number = 0

        if self.record_global_position and hasattr(self.env, "width"):
            self.record["Env Width"] = []
            self.record["Env Length"] = []
            self.record_dims = True
        else:
            self.record_dims = False

    def _distance_to_crosswalk(self, positions: torch.Tensor):
        return -positions[:, 0]

    def _distance_to_pedestrians(
        self,
        positions: torch.Tensor,  # N x 2
        theta: torch.Tensor,  # N x 1
        pedestrians: torch.Tensor,  # P x 2
    ):
        positions = positions.unsqueeze(1)  # N x 1 x 2
        pedestrians = pedestrians.unsqueeze(0)  # 1 x P x 2
        vec_ = pedestrians - positions  # N x P x 2
        vec = vec_ / (torch.norm(vec_, dim=2, keepdim=True) + 1e-7)
        phi = torch.atan2(vec[..., 1:], vec[..., :1])  # N x P x 1
        theta = torch.where(theta >= 0, theta, theta + 2 * math.pi).unsqueeze(
            1
        )  # N x 1 x 1
        diff = phi - theta  # N x P x 1
        angle = angle_normalize(diff.view(-1, 1)).view(vec_.shape[:2])  # N x P

        distances = vec_.pow(2).sum(dim=2).sqrt()

        visible = (
            (angle.abs() > math.pi / 6) + (distances <= 0.5)
        ) * 1e12 + distances

        _, idxs = visible.min(1)

        values = []
        for i in range(positions.shape[0]):
            idx = idxs[i]
            values.append(visible[i : (i + 1), idx])
        return torch.cat(values)

    def _distance_to_nearest_car(
        self, positions: torch.Tensor, theta: torch.Tensor, vel: torch.Tensor
    ):
        # The nearest car needs to be within a conical section in the front
        points = positions.unsqueeze(1).repeat(1, positions.size(0), 1)
        vec_ = positions - points
        vec = vec_ / (torch.norm(vec_, dim=2, keepdim=True) + 1e-7)
        phi = torch.atan2(vec[..., 1:], vec[..., :1])
        theta = torch.where(theta >= 0, theta, theta + 2 * math.pi).unsqueeze(
            1
        )
        diff = phi - theta
        angle = angle_normalize(diff.view(-1, 1)).view(points.shape[:2])

        distances = vec_.pow(2).sum(dim=2).sqrt()

        visible = (
            (angle.abs() > math.pi / 6) + (distances <= 0.5)
        ) * 1e12 + distances

        _, idxs = visible.min(1)

        values = []
        rel_vels = []
        for i in range(positions.shape[0]):
            idx = idxs[i]
            rel_vels.append(vel[i : (i + 1), 0] - vel[idx : (idx + 1), 0])
            values.append(visible[i : (i + 1), idx])
        return torch.cat(values), torch.cat(rel_vels)

    def _action_observation_hook(
        self, action, observation, aids, *args, **kwargs
    ):
        if len(args) == 1 and args[0] == 0:
            return
        state = self.env.world.get_all_vehicle_state()
        observation = (
            observation
            if not isinstance(observation, (tuple, list))
            else observation[0]
        )
        ts = observation[:, -4 if not "Communication" in self.env_name else -5]
        heading = self.env.agents["agent"].optimal_heading()
        if self.record_accln_rating:
            rating = self.env.accln_rating
        if self.record_recv_communication:
            comm = self.env.world.comm_channel[0]
        if self.record_send_communication:
            comm_data = self.env.world.get_broadcast_data_all_agents()
        if self.record_min_distance_to_car:
            distances, rel_vels = self._distance_to_nearest_car(
                state[:, :2], state[:, 3:], state[:, 2:3]
            )
        if self.record_distance_to_crosswalk:
            dcrosswalk = self._distance_to_crosswalk(state[:, :2])
        if self.record_pedestrian_distance:
            dpedestrian = self._distance_to_pedestrians(
                state[:, :2],
                state[:, 3:],
                self.env.world.objects["pedestrian"].position,
            )

        for i in range(action.size(0)):
            if self.record_traffic_signal:
                self.record["Traffic Signal"].append(ts[i].item())
            self.record["Velocity"].append(state[i, 2].item())
            self.record["Acceleration"].append(action[i, -1].item())
            self.record["Time Step"].append(self.timesteps[i])
            self.record["Episode"].append(self.episode_number)
            self.record["Agent ID"].append(aids[i])
            self.record["Heading"].append(heading[i, 0].item())
            if self.record_steering:
                self.record["Steering Angle"].append(action[i, 0].item())
            if self.record_global_position:
                self.record["Position"].append(
                    state[i, :2].cpu().numpy().tolist()
                )
                if self.record_dims:
                    self.record["Env Width"].append(self.env.width)
                    self.record["Env Length"].append(self.env.length)
            if self.record_accln_rating:
                self.record["Acceleration Rating"].append(rating[i, 0].item())
            if self.record_send_communication:
                self.record["Communication (Send)"].append(comm[i].numpy())
            if self.record_recv_communication:
                self.record["Communication (Recv)"].append(
                    comm_data[i].numpy()
                )
            if self.record_min_distance_to_car:
                self.record["Minimum Distance to Car"].append(
                    distances[i].item()
                )
                self.record["Relative Velocity"].append(rel_vels[i].item())
            if self.record_distance_to_crosswalk:
                self.record["Distance to Crosswalk"].append(
                    dcrosswalk[i].item()
                )
            if self.record_pedestrian_distance:
                self.record["Distance to Nearest Pedestrian"].append(
                    dpedestrian[i].item()
                )
            self.timesteps[i] += 1

    def _new_rollout_hook(self):
        self.timesteps = [0] * self.env.nagents
        self.episode_number += 1

    def _post_completion_hook(self):
        df = pd.DataFrame.from_dict(self.record)
        df.to_csv(str(self.fname))

        print(f"Saved DataFrame to {self.fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-dir", type=str, required=True)
    parser.add_argument("-f", "--fname", type=str, required=True)
    parser.add_argument("-m", "--model-save-path", type=str, default=None)
    parser.add_argument("-tep", "--num-test-episodes", type=int, default=1)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument(
        "--model-type", default=None, choices=["one_step", "two_step", None]
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simulator = RolloutSimulatorActionRecorder(
        args.fname,
        args.env,
        args.env_kwargs,
        device,
        args.save_dir,
        args.model_save_path,
        args.model_type,
    )

    simulator.rollout(args.num_test_episodes, args.verbose, not args.no_render)
