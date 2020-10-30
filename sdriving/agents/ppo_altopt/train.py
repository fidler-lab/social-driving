import argparse
import json
import os

import gym
import horovod.torch as hvd

from sdriving.agents.ppo_altopt.ppo import (
    PPO_Alternating_Optimization_Centralized_Critic,
)
from sdriving.environments import REGISTRY as ENV_REGISTRY

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eid", required=True, type=str)
    parser.add_argument("-s", "--save-dir", required=True, type=str)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--vf-lr", type=float, default=1e-3)
    parser.add_argument("--pi-lr", type=float, default=3e-4)
    parser.add_argument("--spline-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=1e-2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("-se1", "--steps-per-epoch1", type=int, default=4000)
    parser.add_argument("-se2", "--steps-per-epoch2", type=int, default=4000)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("-tep", "--num-test-episodes", type=int, default=1)
    parser.add_argument("-ti", "--train_iters", type=int, default=80)
    parser.add_argument("--actor-kwargs", type=json.loads, default={})
    parser.add_argument("--ac-kwargs", type=json.loads, default={})
    parser.add_argument("--logger-kwargs", type=json.loads, default={})
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-checkpoint", type=str, default="")
    parser.add_argument("-wid", "--wandb-id", type=str, default=None)

    args = parser.parse_args()

    hvd.init()

    env = ENV_REGISTRY[args.env]
    log_dir = os.path.join(args.save_dir, args.eid)

    trainer = PPO_Alternating_Optimization_Centralized_Critic(
        env,
        args.env_kwargs,
        log_dir,
        actor_kwargs=args.actor_kwargs,
        ac_kwargs=args.ac_kwargs,
        seed=args.seed,
        number_episodes_per_spline_update=args.steps_per_epoch1,
        number_steps_per_controller_update=args.steps_per_epoch2,
        epochs=args.epochs,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        spline_lr=args.spline_lr,
        train_iters=args.train_iters,
        lam=args.lam,
        target_kl=args.target_kl,
        load_path=args.model_checkpoint if args.resume else None,
        wandb_id=args.wandb_id,
        entropy_coeff=args.entropy_coeff,
    )

    trainer.train()
