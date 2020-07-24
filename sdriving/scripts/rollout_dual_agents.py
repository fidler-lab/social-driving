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
    parser.add_argument("--algo", default="PPO", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ENV_REGISTRY[args.env](**args.env_kwargs)

    if not args.dummy_run:
        actor, ac = checkpoint_parser(args.model_save_path)
        ac = ac.to(device)
        actor = actor.to(device)

    total_ret = 0.0
    for ep in range(args.num_test_episodes):
        ep_ret, ep_len = 0, 0
        env.reset()
        # Spline Action
        a = {}
        for a_id, obs in env.get_spline_state().items():
            a[a_id] = (
                torch.as_tensor(env.spline_action_space.sample())
                if args.dummy_run
                else actor.act(obs.to(device), True).cpu()
            )
        env.spline_step(a)
        # Controller Action
        done = False
        o = env.get_controller_state()
        while not done:
            a = {}
            for a_id, obs in o.items():
                a[a_id] = (
                    torch.as_tensor(env.controller_action_space.sample())
                    if args.dummy_run
                    else ac.act([o1.to(device) for o1 in obs], True).cpu()
                )
            o, r, d, _ = env.controller_step(
                a,
                render=not args.no_render,
                lims={"x": (-100.0, 100.0), "y": (-100.0, 100.0)},
            )

            rlist = [
                torch.as_tensor(rwd).detach().cpu() for _, rwd in r.items()
            ]
            ret = sum(rlist)
            rlen = len(rlist)
            ep_ret += ret / rlen
            ep_len += 1

            done = d["__all__"]
            if args.verbose:
                print(f"Reward: {r}")
        total_ret += ep_ret
        print(
            f"Episode {ep} : Total Length: {ep_len} | Total Return: {ep_ret}"
        )
        if not args.no_render:
            path = os.path.join(args.save_dir, f"test_{ep}.mp4")
            env.render(path=path)
            print(f"Episode saved at {path}")
    print(
        f"Mean Return over {args.num_test_episodes} episodes: "
        + f"{total_ret / args.num_test_episodes}"
    )
