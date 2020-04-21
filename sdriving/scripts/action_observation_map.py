import argparse
import json
import logging
import os
import random
import sys
import time

import gym
import numpy as np
import pandas as pd
import torch

from sdriving.envs import REGISTRY as ENV_REGISTRY


# Handles the irritating convergence message from mpc pytorch
class CustomPrint:
    def __init__(self):
        self.old_stdout = sys.stdout

    def write(self, text):
        text = text.rstrip()
        if len(text) == 0:
            return
        if "pnqp warning" in text:
            return
        self.old_stdout.write(text + "\n")

    def flush(self):
        self.old_stdout.flush()


# Not a big fan of doing this, but don't know any other way to
# handle it
sys.stdout = CustomPrint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-path", type=str, required=True)
    parser.add_argument("-m", "--model-save-path", type=str)
    parser.add_argument("-tep", "--num-test-episodes", type=int, default=1)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--dummy-run", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.dummy_run:
        ckpt = torch.load(args.model_save_path, map_location="cpu")
        if "model" not in ckpt or ckpt["model"] == "centralized_critic":
            from sdriving.agents.ppo_cent.model import ActorCritic
        ac = ActorCritic(**ckpt["ac_kwargs"])
        ac.v = None
        ac.pi.load_state_dict(ckpt["actor"])
        ac = ac.to(device)

    test_env = ENV_REGISTRY[args.env](**args.env_kwargs)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    df = dict()
    # FIXME: Generalize for other gym spaces
    # Right now only handle Tuple
    if args.env in (
        "RoadIntersectionControlEnv",
        "RoadIntersectionControlAccelerationEnv",
        "RoadIntersectionControlImitateEnv",
    ):
        df["Traffic Signal"] = []
        df["Velocity"] = []
        # This is the distance from signal only when signal is visible
        df["Distance from Signal"] = []
        df["Agent ID"] = []
        df["Acceleration"] = []
        df["Steering Angle"] = []

    total_ret = 0.0
    count = 0
    for ep in range(args.num_test_episodes):
        o, done, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not done:
            # Take deterministic actions at test time
            a = {}
            for key, obs in o.items():
                df["Agent ID"].append(key)
                if isinstance(obs, tuple):
                    obs = [t.to(device) for t in obs]
                    df["Traffic Signal"].append(obs[0][-4].item())
                    df["Velocity"].append(-obs[0][-3].item() * 2 * 8.0)
                    df["Distance from Signal"].append((1 / obs[0][-1]).item())
                else:
                    obs = obs.to(device)
                o[key] = obs
                if not args.dummy_run:
                    a[key] = ac.act(obs, True)
                else:
                    a[key] = torch.as_tensor(test_env.action_space.sample())
                df["Steering Angle"].append(
                    test_env.actions_list[a[key]][0, 0].item()
                )
                df["Acceleration"].append(
                    test_env.actions_list[a[key]][0, 1].item()
                )
                if args.verbose:
                    print(
                        f"Agent: {key} || Observation: {obs[0][:-4]} || Action: {a[key]}"
                    )
            pts = {}
            o, r, d, _ = test_env.step(
                a,
                render=not args.no_render,
                pts=pts,
                lims={"x": (-100.0, 100.0), "y": (-100.0, 100.0)},
            )
            ep_ret += sum([rwd for _, rwd in r.items()])
            ep_len += 1
            done = d["__all__"]
            if args.verbose:
                print(f"Reward: {sum(r.values())}")
        total_ret += ep_ret
        print(
            f"Episode {ep} : Total Length: {ep_len} | Total Return: {ep_ret}"
        )
        if ep_ret < 0.0:
            count += 1
        if not args.no_render:
            path = os.path.join(
                os.path.dirname(args.save_path), f"test_{ep}.mp4"
            )
            test_env.render(path=path)
            print(f"Episode saved at {path}")
    print(
        f"Mean Return over {args.num_test_episodes} episodes: "
        + f"{total_ret / args.num_test_episodes}"
    )
    print(
        f"Crashes = {count} | Success Rate = {(1 - count / args.num_test_episodes) * 100}"
    )

    df = pd.DataFrame.from_dict(df)
    df.to_csv(args.save_path)
