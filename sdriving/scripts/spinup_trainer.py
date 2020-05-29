import argparse
import json
import logging
import os

from sdriving.envs.gym import REGISTRY as ENV_REGISTRY
from spinup import ppo_pytorch
from spinup.utils.mpi_tools import mpi_fork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(process)d:%(message)s",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eid", required=True, type=str)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("-rs", "--replay-size", type=int, default=int(1e6))
    parser.add_argument("--vf-lr", type=float, default=1e-3)
    parser.add_argument("--pi-lr", type=float, default=3e-4)
    parser.add_argument("-s", "--save-dir", required=True, type=str)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("-se", "--steps-per-epoch", type=int, default=4000)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("-tep", "--num-test-episodes", type=int, default=1)
    parser.add_argument("-tpi", "--train_pi_iters", type=int, default=10)
    parser.add_argument("-tvi", "--train_v_iters", type=int, default=10)
    parser.add_argument("-f", "--save-freq", type=int, default=1)
    parser.add_argument("--ac-kwargs", type=json.loads, default={})
    parser.add_argument("--logger-kwargs", type=json.loads, default={})

    args = parser.parse_args()

    env = ENV_REGISTRY[args.env]
    log_dir = os.path.join(args.save_dir, args.eid)

    mpi_fork(args.cpu)

    lgkwargs = args.logger_kwargs
    lgkwargs["output_dir"] = log_dir

    # NOTE: This doesn't support preemption :'(
    ppo_pytorch(
        lambda: env(args.env_kwargs),
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
        logger_kwargs=lgkwargs,
        save_freq=args.save_freq,
    )
