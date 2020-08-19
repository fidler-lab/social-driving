import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Union

import gym
import numpy as np
import torch
from sdriving.environments import REGISTRY as ENV_REGISTRY
from sdriving.scripts.ckpt_parser import checkpoint_parser


class RolloutSimulator:
    def __init__(
        self,
        env_name: str,
        env_kwargs: dict,
        device: torch.device,
        save_dir: str,
        load_path: Optional[str] = None
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs

        self.env = ENV_REGISTRY[self.env_name](**self.env_kwargs)

        if load_path is not None:
            self.dummy_run = False
            models, tag = checkpoint_parser(load_path)
            if isinstance(models, [list, tuple]):
                self.ac, self.actor = models
                self.ac.to(device)
                self.actor.to(device)
                self.two_stage_rollout = True
            else:
                self.actor = models.to(device)
                self.two_stage_rollout = False
        else:
            self.dummy_run = True

        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    @torch.no_grad()
    def rollout(self, nepisodes: int, verbose: bool = False, render: bool = True):
        for ep in range(nepisodes):
            total_ret = 0

            if self.two_stage_rollout:
                ep_ret, ep_len = self._two_stage_rollout(verbose, render)
            else:
                ep_ret, ep_len = self._one_stage_rollout(verbose, render)

            total_ret += ep_ret

            print(f"Episode: {ep} | Length: {ep_len} | Return: {ep_ret}")

            if render:
                path = self.save_dir / f"{self.env_name}_{ep}.mp4"
                self.env.render(path=path)
                print("Episode Render saved at {path}")

        print(f"Mean Return over {nepisodes} episodes:"
              f" {total_ret / nepisodes}")

    def _move_object_to_device(self, obj: Union[tuple, list, torch.Tensor]):
        if isinstance(obj, [list, tuple]):
            return [o.to(self.device) for o in obj]
        return obj.to(self.device)

    @torch.no_grad()
    def _action_one_stage_rollout(self, obs):
        if self.dummy_run:
            return torch.cat([
                torch.as_tensor(self.env.action_space.sample()).unsqueeze(0)
                for _ in range(self.env.nagents)
            ]).cpu()

        obs = self._move_object_to_device(obs)
        return self.actor.act(obs, deterministic=True).cpu()

    @torch.no_grad()
    def _one_stage_rollout(self, verbose: bool, render: bool):
        o, done, ep_ret, ep_len = self.env.reset(), False, 0, 0

        while not done:
            a = self._action_one_stage_rollout(o)

            if verbose:
                print(f"Observation: {o}")
                print(f"Action: {a}")

            o, r, d, _ = self.env.step(
                a,
                render=render,
                lims={"x": (-100.0, 100.0), "y": (-100.0, 100.0)}
            )

            ep_ret = ep_ret + r
            ep_len += 1

            done = d.all()

            if verbose:
                print(f"Reward: {r.mean()}")

        ep_ret = ep_ret.mean()
        return ep_ret, ep_len

    @torch.no_grad()
    def _action_two_stage_rollout(self, stage:int, obs):
        if self.dummy_run:
            return torch.cat([
                torch.as_tensor(self.env.action_space[stage].sample()).unsqueeze(0)
                for _ in range(self.env.nagents)
            ]).cpu()

        obs = self._move_object_to_device(obs)
        model = self.actor if stage == 0 else self.ac
        return model.act(obs, deterministic=True).cpu()

    @torch.no_grad()
    def _two_stage_rollout(self, verbose: bool, render: bool):
        o, done, ep_ret, ep_len = self.env.reset(), False, 0, 0

        a = self._action_two_stage_rollout(0, o)
        o = self.env.step(0, a)

        while not done:
            a = self._action_two_stage_rollout(o)

            if verbose:
                print(f"Observation: {o}")
                print(f"Action: {a}")

            o, r, d, _ = self.env.step(
                1,
                a,
                render=render,
                lims={"x": (-100.0, 100.0), "y": (-100.0, 100.0)}
            )

            ep_ret = ep_ret + r
            ep_len += 1

            done = d.all()

            if verbose:
                print(f"Reward: {r.mean()}")

        ep_ret = ep_ret.mean()
        return ep_ret, ep_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-dir", type=str, required=True)
    parser.add_argument("-m", "--model-save-path", type=str, default=None)
    parser.add_argument("-tep", "--num-test-episodes", type=int, default=1)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--dummy-run", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simulator = RolloutSimulator(
        args.env, args.env_kwargs, device, args.save_dir, args.model_save_path
    )

    simulator.rollout(args.num_test_episodes, args.verbose, not args.no_render)

