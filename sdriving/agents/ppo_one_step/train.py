import argparse
import json
import logging
import os

import gym

from sdriving.agents.ppo_one_step.ppo import PPO_OneStep
from sdriving.environments import REGISTRY as ENV_REGISTRY

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
    parser.add_argument("--entropy-coeff", type=float, default=1e-2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("-se", "--steps-per-epoch", type=int, default=4000)
    parser.add_argument("-tpi", "--train_pi_iters", type=int, default=1)
    parser.add_argument("-f", "--save-freq", type=int, default=1)
    parser.add_argument("--actor-kwargs", type=json.loads, default={})
    parser.add_argument("--logger-kwargs", type=json.loads, default={})
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-checkpoint", type=str, default="")
    parser.add_argument("-wid", "--wandb-id", type=str, default=None)

    args = parser.parse_args()

    env = ENV_REGISTRY[args.env]
    log_dir = os.path.join(args.save_dir, args.eid)

    trainer = PPO_OneStep(
        env,
        args.env_kwargs,
        log_dir,
        actor_kwargs=args.actor_kwargs,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        clip_ratio=args.clip_ratio,
        epochs=args.epochs,
        pi_lr=args.pi_lr,
        train_pi_iters=args.train_pi_iters,
        entropy_coeff=args.entropy_coeff,
        logger_kwargs=args.logger_kwargs,
        wandb_id=args.wandb_id,
        load_path=args.model_checkpoint if args.resume else None,
    )

    trainer.train()
