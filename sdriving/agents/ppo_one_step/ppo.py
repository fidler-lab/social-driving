import os
import time
import warnings
from typing import Optional

from mpi4py import MPI

import gym
import numpy as np
import torch
import wandb
from sdriving.agents.buffer import *
from sdriving.agents.model import *
from sdriving.agents.utils import (
    count_vars,
    mpi_avg_grads,
    trainable_parameters,
)
from sdriving.agents.ppo_one_step.runner import episode_runner
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from spinup.utils.mpi_tools import (
    mpi_avg,
    mpi_fork,
    mpi_statistics_scalar,
    num_procs,
    proc_id,
    mpi_op,
)
from spinup.utils.serialization_utils import convert_json
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter


class PPO_OneStep:
    def __init__(
        self,
        env,
        env_params: dict,
        log_dir: str,
        actor_kwargs: dict = {},
        seed: int = 0,
        steps_per_epoch: int = 4000,
        epochs: int = 50,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        train_pi_iters: int = 80,
        entropy_coeff: float = 1e-2,
        target_kl: float = 0.01,
        logger_kwargs: dict = {},
        save_freq: int = 10,
        load_path=None,
        render_train: bool = False,
        wandb_id: Optional[str] = None,  # Optional exists for legacy code
        **kwargs,
    ):
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Set up logger and save configuration
        self.log_dir = os.path.join(log_dir, str(proc_id()))
        hparams = convert_json(locals())
        self.logger = EpochLogger(log_dir, **logger_kwargs)
        self.render_dir = os.path.join(log_dir, "renders")
        os.makedirs(self.render_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.softlink = os.path.abspath(
            os.path.join(self.ckpt_dir, f"ckpt_latest.pth")
        )

        self.logger.save_config(locals())

        self.env = env(**env_params)
        self.actor_params = {k: v for k, v in actor_kwargs.items()}
        self.actor_params.update(
            {
                "obs_dim": self.env.observation_space.shape[0],
                "act_space": self.env.action_space,
            }
        )

        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff / epochs

        if torch.cuda.is_available():
            # From emperical results, 8 tasks can use a single gpu
            device_id = proc_id() // 8
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")

        if os.path.isfile(self.softlink):
            self.logger.log("Restarting from latest checkpoint", color="red")
            load_path = self.softlink

        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        if render_train:
            self.logger.log(
                "Rendering the training is not implemented", color="red"
            )

        self.nagents = self.env.nagents
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.actor = PPOWaypointCategoricalActor(**self.actor_params)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.actor = PPOWaypointGaussianActor(**self.actor_params)

        self.device = device

        self.pi_lr = pi_lr

        self.load_path = load_path
        if load_path is not None:
            self.load_model(load_path)
        else:
            self.pi_optimizer = Adam(
                trainable_parameters(self.actor), lr=self.pi_lr, eps=1e-8
            )

        # Sync params across processes
        sync_params(self.actor)
        self.actor = self.actor.to(device)

        if proc_id() == 0:
            if wandb_id is None:
                eid = (
                    log_dir.split("/")[-2]
                    if load_path is None
                    else load_path.split("/")[-4]
                )
            else:
                eid = wandb_id
            wandb.init(
                name=eid,
                id=eid,
                project="Social Driving",
                resume=load_path is not None,
                allow_val_change=True,
            )
            wandb.watch_called = False

            if "self" in hparams:
                del hparams["self"]
            wandb.config.update(hparams, allow_val_change=True)

            wandb.watch(self.actor, log="all")

        # Count variables
        var_counts = count_vars(self.actor)
        self.logger.log(f"\nNumber of parameters: \t pi: {var_counts}\n")

        # Set up experience buffer
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = OneStepPPOBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            self.local_steps_per_epoch,
            self.env.nagents,
        )

        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.target_kl = target_kl
        self.epochs = epochs
        self.save_freq = save_freq
        
    def compute_loss(self, data: dict, idx):
        device = self.device
        clip_ratio = self.clip_ratio

        # Random subset sampling
        data = {key: val[idx] for key, val in data.items()}
        obs, rew, act, logp_old = map(
            lambda x: data[x].to(device), ["obs", "rew", "act", "logp"],
        )
        rew = (rew - rew.mean()) / (rew.std() + 1e-7)

        # Policy loss
        pi, logp = self.actor(obs, act)
        ratio = torch.exp((logp - logp_old).clamp(-20.0, 2.0))
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * rew
        loss_pi = -(torch.min(ratio * rew, clip_adv)).mean()

        # Entropy Loss
        ent = pi.entropy().mean()

        # TODO: Search for a good set of coeffs
        loss = loss_pi - ent * self.entropy_coeff

        # Logging Utilities
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        info = dict(
            kl=approx_kl,
            ent=ent.item(),
            cf=clipfrac,
            pi_loss=loss_pi.item(),
        )

        return loss, info
    
    def update(self, lspe=None):
        data = self.buf.get()
        local_steps_per_epoch = (
            self.local_steps_per_epoch if lspe is None else lspe
        )

        train_iters = self.train_pi_iters
        size = int(mpi_op(data["obs"].size(0), MPI.MIN))
        batch_size = size // train_iters

        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(range(size)),
            batch_size,
            drop_last=True,
        )

        with torch.no_grad():
            _, info = self.compute_loss(data, range(size))
            pi_l_old = info["pi_loss"]

        for i, idx in enumerate(sampler):
            self.pi_optimizer.zero_grad()

            loss, info = self.compute_loss(data, idx)

            kl = mpi_avg(info["kl"])
            if kl > 1.5 * self.target_kl:
                self.logger.log(
                    f"Early stopping at step {i} due to reaching max kl."
                )
                break
            loss.backward()

            self.actor = self.actor.cpu()
            mpi_avg_grads(self.actor)
            self.actor = self.actor.to(self.device)

            self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Log changes from update
        kl, ent, cf = info["kl"], info["ent"], info["cf"]
        self.logger.store(
            LossPi=pi_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(info["pi_loss"] - pi_l_old),
        )
        if proc_id() == 0:
            wandb.log(
                {
                    "Loss Actor": pi_l_old,
                    "KL Divergence": kl,
                    "Entropy": ent,
                    "Clip Factor": cf,
                }
            )
            
    def save_model(self, epoch: int, ckpt_extra: dict = {}):
        ckpt = {
            "actor": self.actor.state_dict(),
            "nagents": self.nagents,
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "actor_kwargs": self.actor_params,
            "model": "centralized_critic",
            "type": "one_step_ppo",
        }
        ckpt.update(ckpt_extra)
        torch.save(ckpt, self.softlink)
        wandb.save(self.softlink)

    def load_model(self, load_path):
        ckpt = torch.load(load_path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.pi_optimizer = Adam(
            trainable_parameters(self.actor), lr=self.pi_lr, eps=1e-8
        )
        self.pi_optimizer.load_state_dict(ckpt["pi_optimizer"])
        for state in self.pi_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
                    
    def dump_logs(self, epoch, start_time):
        self.logger.log_tabular("Epoch", epoch)
        self.logger.log_tabular("EpRet", with_min_and_max=True)
        self.logger.log_tabular(
            "TotalEnvInteracts", (epoch + 1) * self.steps_per_epoch
        )
        self.logger.log_tabular("LossPi", average_only=True)
        self.logger.log_tabular("DeltaLossPi", average_only=True)
        self.logger.log_tabular("Entropy", average_only=True)
        self.logger.log_tabular("KL", average_only=True)
        self.logger.log_tabular("ClipFrac", average_only=True)
        self.logger.log_tabular("StopIter", average_only=True)
        self.logger.log_tabular("Time", time.time() - start_time)
        self.logger.dump_tabular()
        
    def train(self):
        # Prepare for interaction with environment
        start_time = time.time()

        for epoch in range(self.epochs):
            episode_runner(
                self.local_steps_per_epoch,
                self.device,
                self.buf,
                self.env,
                self.actor,
                self.logger,
            )

            if (
                (epoch % self.save_freq == 0) or (epoch == self.epochs - 1)
            ) and proc_id() == 0:
                self.save_model(epoch)

            self.update()

            # Log info about epoch
            self.dump_logs(epoch, start_time)