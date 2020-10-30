import os
import time
from typing import Optional

import gym
import horovod.torch as hvd
import numpy as np
import torch
import wandb
from torch.optim import Adam

from sdriving.agents.buffer import OneStepPPOBuffer
from sdriving.agents.model import (
    PPOWaypointCategoricalActor,
    PPOWaypointGaussianActor,
)
from sdriving.agents.utils import (
    count_vars,
    hvd_average_grad,
    hvd_scalar_statistics,
    trainable_parameters,
)
from sdriving.logging import EpochLogger, convert_json


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
        wandb_id: Optional[str] = None,
        **kwargs,
    ):
        self.log_dir = log_dir
        self.render_dir = os.path.join(log_dir, "renders")
        self.ckpt_dir = os.path.join(log_dir, "checkpoints")
        if hvd.rank() == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.render_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.softlink = os.path.abspath(
            os.path.join(self.ckpt_dir, f"ckpt_latest.pth")
        )
        self.ac_params_file = os.path.join(log_dir, "ac_params.json")
        hparams = convert_json(locals())
        self.logger = EpochLogger(output_dir=self.log_dir, exp_name=wandb_id)

        if torch.cuda.is_available():
            # Horovod: pin GPU to local rank.
            dev_id = int(
                torch.cuda.device_count() * hvd.local_rank() / hvd.local_size()
            )
            torch.cuda.set_device(dev_id)
            device = torch.device(f"cuda:{dev_id}")
            torch.cuda.manual_seed(seed)
        else:
            device = torch.device("cpu")

        #         env_params.update({"device": device})
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

        # Horovod: limit # of CPU threads to be used per worker.
        torch.set_num_threads(1)

        torch.save(self.ac_params, self.ac_params_file)

        if os.path.isfile(self.softlink):
            self.logger.log("Restarting from latest checkpoint", color="red")
            load_path = self.softlink

        # Random seed
        seed += 10000 * hvd.rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

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
        hvd.broadcast_parameters(self.ac.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.pi_optimizer, root_rank=0)
        self.actor = self.actor.to(device)
        self.move_optimizer_to_device(self.pi_optimizer)

        if hvd.rank() == 0:
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
        # Set up experience buffer
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = int(steps_per_epoch / hvd.size())
        self.buf = OneStepPPOBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            self.local_steps_per_epoch,
            self.env.nagents,
            device,
        )

        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.target_kl = target_kl
        self.epochs = epochs
        self.save_freq = save_freq

    def compute_loss(self, data):
        self.device
        clip_ratio = self.clip_ratio

        (
            obs,
            act,
            logp_old,
            rew,
        ) = [data[k] for k in ["obs", "act", "logp", "rew"]]

        # Policy loss
        pi, _, logp = self.actor(obs, act)
        ratio = torch.exp(logp - logp_old)  # N x B
        clip_rew = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * rew
        loss_pi = -torch.min(ratio * rew, clip_rew).mean()

        # Entropy Loss
        ent = pi.entropy().mean()

        loss = loss_pi - ent * self.entropy_coeff
        self.entropy_coeff -= self.entropy_coeff_decay

        # Logging Utilities
        approx_kl = (logp_old - logp).mean().detach().cpu()
        info = dict(
            kl=approx_kl,
            ent=ent.item(),
            pi_loss=loss_pi.item(),
        )

        return loss, info

    def update(self):
        data = self.buf.get()
        self.local_steps_per_epoch
        self.train_iters

        for i in range(train_pi_iters):
            self.pi_optimizer.zero_grad()

            loss, info = self.compute_loss(data)

            kl = hvd.allreduce(info["kl"], op=hvd.Average)
            loss.backward()
            hvd_average_grad(self.actor, self.device)
            if kl > 1.5 * self.target_kl:
                self.logger.log(
                    f"Early stopping at step {i} due to reaching max kl.",
                    color="red",
                )
                break
            self.pi_optimizer.step()
        self.logger.store(StopIter=i)

        # Log changes from update
        ent, pi_l_old = info["ent"], info["pi_loss"]
        self.logger.store(
            LossActor=pi_l_old,
            KL=kl,
            Entropy=ent,
        )

    def move_optimizer_to_device(self, opt):
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

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

    def dump_tabular(self):
        self.logger.log_tabular("Epoch", average_only=True)
        self.logger.log_tabular("EpisodeReturn", with_min_and_max=True)
        self.logger.log_tabular("Entropy", average_only=True)
        self.logger.log_tabular("KL", average_only=True)
        self.logger.log_tabular("StopIter", average_only=True)
        self.logger.log_tabular("LossActor", average_only=True)
        self.logger.log_tabular("EpisodeRunTime", average_only=True)
        self.logger.log_tabular("PPOUpdateTime", average_only=True)
        self.logger.dump_tabular()

    def train(self):
        # Prepare for interaction with environment
        for epoch in range(self.epochs):
            self.logger.store(Epoch=epoch)

            start_time = time.time()
            self.episode_runner()
            self.logger.store(EpisodeRunTime=time.time() - start_time)
            if (
                (epoch % self.save_freq == 0) or (epoch == self.epochs - 1)
            ) and hvd.rank() == 0:
                self.save_model(epoch)

            start_time = time.time()
            self.update()
            self.logger.store(PPOUpdateTime=time.time() - start_time)
            self.logger.store(Epoch=epoch)

            self.dump_tabular()

    def episode_runner(self):
        env = self.env

        for t in range(self.local_steps_per_epoch):
            (o, a_ids) = env.reset()
            _, actions, log_probs = self.actor(o.to(self.device))
            _, r, _, _ = env.step(actions)
            ep_ret = r.mean()

            self.buf.store(
                o.detach(),
                actions.detach(),
                r[:, 0].to(self.device).detach(),
                log_probs.detach(),
            )
            self.logger.store(EpisodeReturn=ep_ret)
