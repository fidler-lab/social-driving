import argparse
import json
import logging
import os
import random
import sys
import time

import gym
import numpy as np
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
    parser.add_argument("-s", "--save-dir", type=str, required=True)
    parser.add_argument("-m", "--model-save-path", type=str)
    parser.add_argument("-tep", "--num-test-episodes", type=int, default=1)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--dummy-run", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--algo", default="PPO", type=str)
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

    os.makedirs(args.save_dir, exist_ok=True)

    total_ret = 0.0
    for ep in range(args.num_test_episodes):
        o, done, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not done:
            # Take deterministic actions at test time
            a = {}
            for key, obs in o.items():
                if isinstance(obs, tuple):
                    obs = [t.to(device) for t in obs]
                else:
                    obs = obs.to(device)
                o[key] = obs
                if not args.dummy_run:
                    a[key] = ac.act(obs, True)
                else:
                    a[key] = torch.as_tensor(test_env.action_space.sample())
                if args.verbose:
                    print(
                        f"Agent: {key} || Observation: {obs[0][-4:]} || Action: {a[key]}"
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
        if not args.no_render:
            path = os.path.join(args.save_dir, f"test_{ep}.mp4")
            test_env.render(path=path)
            print(f"Episode saved at {path}")
    print(
        f"Mean Return over {args.num_test_episodes} episodes: "
        + f"{total_ret / args.num_test_episodes}"
    )
