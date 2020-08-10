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
from sdriving.environments import REGISTRY as ENV_REGISTRY
from sdriving.scripts.ckpt_parser import checkpoint_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-dir", type=str, required=True)
    parser.add_argument("-m", "--model-save-path", type=str)
    parser.add_argument("-tep", "--num-test-episodes", type=int, default=1)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--dummy-run", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_env = ENV_REGISTRY[args.env](**args.env_kwargs)

    if not args.dummy_run:
        ac = checkpoint_parser(args.model_save_path)
        ac = ac.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        total_ret = 0.0
        for ep in range(args.num_test_episodes):
            o, done, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not done:
                # Take deterministic actions at test time
                if not args.dummy_run:
                    if isinstance(o, list) or isinstance(o, tuple):
                        o = [obs.to(device) for obs in o]
                    else:
                        o = o.to(device)
                    a = ac.act(o, True).cpu()
                if args.verbose:
                    print(f"Observation: {o[0]} || Action: {a}")
                o, r, d, _ = test_env.step(
                    a,
                    render=not args.no_render,
                    lims={"x": (-100.0, 100.0), "y": (-100.0, 100.0)},
                )
                ep_ret = ep_ret + r
                ep_len += 1
                done = d.all()
                if args.verbose:
                    print(f"Reward: {r.mean()}")
            ep_ret = ep_ret.mean()
            total_ret += ep_ret.mean()
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
