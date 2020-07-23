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
from sdriving.agents.alt_opt.ppo import PPO_Alternating_Optimization
from spinup.utils.mpi_tools import mpi_fork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(process)d:%(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eid", required=True, type=str)
    parser.add_argument("-s", "--save-dir", required=True, type=str)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--pi-lr", type=float, default=3e-4)
    parser.add_argument("--vf-lr", type=float, default=3e-4)
    parser.add_argument("--spline-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=1e-2)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument(
        "-sec", "--steps-per-epoch_controller", type=int, default=16000
    )
    parser.add_argument(
        "-ses", "--steps-per-epoch_spline", type=int, default=1600
    )
    parser.add_argument("-ex", "--extra-iters", type=int, default=5)
    parser.add_argument("-tpi", "--train_pi_iters", type=int, default=1)
    parser.add_argument("-tvi", "--train_v_iters", type=int, default=1)
    parser.add_argument("-tsi", "--train_spline_iters", type=int, default=1)
    parser.add_argument("-f", "--save-freq", type=int, default=1)
    parser.add_argument("--actor-kwargs", type=json.loads, default={})
    parser.add_argument("--ac-kwargs", type=json.loads, default={})
    parser.add_argument("--logger-kwargs", type=json.loads, default={})
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-checkpoint", type=str, default="")
    parser.add_argument("-wid", "--wandb-id", type=str, default=None)

    args = parser.parse_args()

    env = ENV_REGISTRY[args.env]
    log_dir = os.path.join(args.save_dir, args.eid)

    mpi_fork(args.cpu)  # run parallel code with mpi

    trainer = PPO_Alternating_Optimization(
        env,
        args.env_kwargs,
        log_dir,
        actor_kwargs=args.actor_kwargs,
        ac_kwargs=args.ac_kwargs,
        seed=args.seed,
        extra_iters=args.extra_iters,
        steps_per_epoch_controller=args.steps_per_epoch_controller,
        steps_per_epoch_spline=args.steps_per_epoch_spline,
        epochs=args.epochs,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        spline_lr=args.spline_lr,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        train_spline_iters=args.train_spline_iters,
        train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters,
        entropy_coeff=args.entropy_coeff,
        lam=args.lam,
        target_kl=args.target_kl,
        logger_kwargs=args.logger_kwargs,
        wandb_id=args.wandb_id,
        load_path=args.model_checkpoint if args.resume else None,
    )

    trainer.train()
