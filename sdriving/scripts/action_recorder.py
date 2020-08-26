import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Union, List

import gym
import pandas as pd
import numpy as np
import torch
from sdriving.scripts.rollout import RolloutSimulator


env2record = {
    "MultiAgentRoadIntersectionBicycleKinematicsEnvironment": (
        ["Traffic Signal", "Velocity", "Acceleration", "Time Step"]
        + ["Episode", "Agent ID", "Steering Angle", "Position"]
    ),
    "MultiAgentRoadIntersectionBicycleKinematicsDiscreteEnvironment": (
        ["Traffic Signal", "Velocity", "Acceleration", "Time Step"]
        + ["Episode", "Agent ID", "Steering Angle", "Position"]
    ),
    "MultiAgentRoadIntersectionFixedTrackEnvironment": (
        ["Traffic Signal", "Velocity", "Acceleration", "Time Step"]
        + ["Episode", "Agent ID"]
    ),
    "MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment": (
        ["Traffic Signal", "Velocity", "Acceleration", "Time Step"]
        + ["Episode", "Agent ID"]
    ),
    "MultiAgentIntersectionSplineAccelerationDiscreteEnvironment": (
        ["Traffic Signal", "Velocity", "Acceleration", "Time Step"]
        + ["Episode", "Agent ID", "Position"]
    ),
    "MultiAgentNuscenesIntersectionDrivingEnvironment": (
        ["Traffic Signal", "Velocity", "Acceleration", "Time Step"]
        + ["Episode", "Agent ID"]
    ),
    "MultiAgentNuscenesIntersectionDrivingDiscreteEnvironment": (
        ["Traffic Signal", "Velocity", "Acceleration", "Time Step"]
        + ["Episode", "Agent ID"]
    ),
}


class RolloutSimulatorActionRecorder(RolloutSimulator):
    def __init__(self, fname: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fname = self.save_dir / fname
    
        # Parse the record items and check which quantities to store
        # Only negate the extra quatities
        self.record_items = env2record[self.env_name]
        
        self.record_steering = "Steering Angle" in self.record_items
        self.record_global_position = "Position" in self.record_items
        
        self.record = {r: [] for r in self.record_items}
        self.episode_number = 0

        if self.record_global_position:
            self.record["Env Width"] = []
            self.record["Env Length"] = []
        
    def _action_observation_hook(
        self, action, observation, *args, **kwargs
    ):
        if len(args) == 1 and args[0] == 0:
            return
        state = self.env.world.get_all_vehicle_state()
        observation = (
            observation
            if not isinstance(observation, (tuple, list))
            else observation[0]
        )
        ts = observation[:, -4]
        for i in range(action.size(0)):
            self.record["Traffic Signal"].append(ts[i].item())
            self.record["Velocity"].append(state[i, 2].item())
            self.record["Acceleration"].append(action[i, -1].item())
            self.record["Time Step"].append(self.timesteps[i])
            self.record["Episode"].append(self.episode_number)
            self.record["Agent ID"].append(i)
            if self.record_steering:
                self.record["Steering Angle"].append(action[i, 0].item())
            if self.record_global_position:
                self.record["Position"].append(state[i, :2].cpu().numpy().tolist())
                self.record["Env Width"].append(self.env.width)
                self.record["Env Length"].append(self.env.length)

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