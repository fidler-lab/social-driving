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
from spinup.utils.mpi_tools import mpi_fork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(process)d:%(message)s",
)


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

    parser.add_argument("--eid", required=True, type=str)
    parser.add_argument("-s", "--save-dir", required=True, type=str)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("-rs", "--replay-size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--vf-lr", type=float, default=1e-3)
    parser.add_argument("--pi-lr", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("-se", "--steps-per-epoch", type=int, default=4000)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("-tep", "--num-test-episodes", type=int, default=1)
    parser.add_argument("-tpi", "--train_pi_iters", type=int, default=80)
    parser.add_argument("-tvi", "--train_v_iters", type=int, default=80)
    parser.add_argument("-f", "--save-freq", type=int, default=1)
    parser.add_argument("--ac-kwargs", type=json.loads, default={})
    parser.add_argument("--logger-kwargs", type=json.loads, default={})
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-train", action="store_true")
    parser.add_argument("--render-test", action="store_true")
    parser.add_argument("--model-checkpoint", type=str, default="")
    parser.add_argument("--tboard", action="store_true")
    parser.add_argument("--centralized-critic", "-cc", action="store_true")

    args = parser.parse_args()

    env = ENV_REGISTRY[args.env]
    log_dir = os.path.join(args.save_dir, args.eid)

    mpi_fork(args.cpu)  # run parallel code with mpi

    test_observation_space = env(**args.env_kwargs).observation_space
    if isinstance(test_observation_space, gym.spaces.Tuple):
        if args.centralized_critic:
            from sdriving.agents.ppo_cent.ppo import (
                PPO_Centralized_Critic as PPO,
            )
        else:
            from sdriving.agents.ppo_indiv.ppo import (
                PPO_Decentralized_Critic as PPO,
            )
    del test_observation_space

    ppo = PPO(
        env,
        args.env_kwargs,
        log_dir,
        ac_kwargs=args.ac_kwargs,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters,
        lam=args.lam,
        target_kl=args.target_kl,
        logger_kwargs=args.logger_kwargs,
        save_freq=args.save_freq,
        load_path=args.model_checkpoint if args.resume else None,
        render_train=args.render_train,
        render_test=args.render_test,
        tboard=args.tboard,
    )

    ppo.train()
