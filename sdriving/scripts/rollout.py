import argparse
import json
from pathlib import Path
from typing import Optional, Union

import gym
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
        load_path: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs

        self.env = ENV_REGISTRY[self.env_name](**self.env_kwargs)

        if load_path is not None:
            self.dummy_run = False
            models, tag = checkpoint_parser(load_path)
            if isinstance(models, (list, tuple)):
                self.ac, self.actor = models
                self.ac.to(device)
                self.actor.to(device)
                self.two_stage_rollout = True
            else:
                self.actor = models.to(device)
                self.two_stage_rollout = False
        else:
            self.dummy_run = True
            if model_type is None:
                raise Exception(
                    "Specify if the model is ['one_step'/'two_step']"
                )
            self.two_stage_rollout = model_type == "two_step"

        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def _action_observation_hook(
        self, action, observation, aids, *args, **kwargs
    ):
        pass

    def _new_rollout_hook(self):
        pass

    def _post_completion_hook(self):
        pass

    @torch.no_grad()
    def rollout(
        self, nepisodes: int, verbose: bool = False, render: bool = True
    ):
        total_ret = 0
        total_crash = 0
        for ep in range(nepisodes):

            if self.two_stage_rollout:
                ep_ret, ep_len, crashed = self._two_stage_rollout(
                    verbose, render
                )
            else:
                ep_ret, ep_len, crashed = self._one_stage_rollout(
                    verbose, render
                )

            total_crash += crashed
            total_ret += ep_ret

            print(
                f"Episode: {ep} | Length: {ep_len} | Return: {ep_ret:0.2f} | Crashed: {crashed.item()}"
            )

            if render:
                path = self.save_dir / f"{self.env_name}_{ep}.mp4"
                self.env.render(path=path)
                print(f"Episode Render saved at {path}")

        print(
            f"Mean Return over {nepisodes} episodes:"
            f" {total_ret / nepisodes} | Total Crashes: {total_crash}"
        )

        self._post_completion_hook()

    def _move_object_to_device(self, obj: Union[tuple, list, torch.Tensor]):
        if isinstance(obj, (list, tuple)):
            return [o.to(self.device) for o in obj]
        return obj.to(self.device)

    @torch.no_grad()
    def _action_one_stage_rollout(self, obs):
        if self.dummy_run:
            return torch.cat(
                [
                    torch.as_tensor(self.env.action_space.sample()).unsqueeze(
                        0
                    )
                    for _ in range(self.env.nagents)
                ]
            ).cpu()
        obs = self._move_object_to_device(obs)
        return self.actor.act(obs, deterministic=True).cpu()

    @torch.no_grad()
    def _one_stage_rollout(self, verbose: bool, render: bool):
        (o, aids), done = self.env.reset(), False
        ep_ret, ep_len = (
            torch.zeros(self.env.nagents, 1, device=self.env.device),
            0,
        )
        self._new_rollout_hook()

        if hasattr(self.env, "accln_rating"):
            print(
                f"Acceleration / Velocity Rating: {self.env.accln_rating[:, 0]}"
            )

        while not done:
            a = self._action_one_stage_rollout(o)

            self._action_observation_hook(
                self.env.discrete_to_continuous_actions(a), o, aids
            )

            if verbose:
                print(f"Observation: {o}")
                print(f"Action: {a}")

            (o, _aids), _r, d, _ = self.env.step(a, render=render)

            r = torch.zeros_like(ep_ret)
            for i, a_id in enumerate(aids):
                b = int(a_id.rsplit("_", 1)[-1])
                r[b] = _r[i]
            aids = _aids

            ep_ret = ep_ret + r
            ep_len += 1

            done = d.all()

            if verbose:
                print(f"Reward: {r.mean()}")

        crashed = (ep_ret < 0).sum()
        ep_ret = ep_ret.mean()
        return ep_ret, ep_len, crashed

    @torch.no_grad()
    def _action_two_stage_rollout(self, stage: int, obs):
        if self.dummy_run:
            return torch.cat(
                [
                    torch.as_tensor(
                        self.env.action_space[stage].sample()
                    ).unsqueeze(0)
                    for _ in range(self.env.nagents)
                ]
            ).cpu()

        obs = self._move_object_to_device(obs)
        model = self.actor if stage == 0 else self.ac
        return model.act(obs, deterministic=True).cpu()

    @torch.no_grad()
    def _two_stage_rollout(self, verbose: bool, render: bool):
        (o, aids), done = self.env.reset(), False
        ep_ret, ep_len = (
            torch.zeros(self.env.nagents, 1, device=self.env.device),
            0,
        )
        self._new_rollout_hook()

        a = self._action_two_stage_rollout(0, o)
        self._action_observation_hook(
            self.env.discrete_to_continuous_actions_v2(a), o, aids, 0
        )

        o, aids = self.env.step(0, a)

        while not done:
            a = self._action_two_stage_rollout(1, o)
            self._action_observation_hook(
                self.env.discrete_to_continuous_actions(a), o, aids, 1
            )

            if verbose:
                print(f"Observation: {o}")
                print(f"Action: {a}")

            (o, _aids), _r, d, _ = self.env.step(1, a, render=render)

            r = torch.zeros_like(ep_ret)
            for i, a_id in enumerate(aids):
                b = int(a_id.rsplit("_", 1)[-1])
                r[b] = _r[i]
            aids = _aids

            ep_ret = ep_ret + r
            ep_len += 1

            done = d.all()

            if verbose:
                print(f"Reward: {r.mean()}")

        crashed = (ep_ret < 0).sum()
        ep_ret = ep_ret.mean()
        return ep_ret, ep_len, crashed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-dir", type=str, required=True)
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

    simulator = RolloutSimulator(
        args.env,
        args.env_kwargs,
        device,
        args.save_dir,
        args.model_save_path,
        args.model_type,
    )

    simulator.rollout(args.num_test_episodes, args.verbose, not args.no_render)
