import os
import time
from typing import Optional

import gym
import horovod.torch as hvd
import numpy as np
import torch
import wandb
from torch.optim import Adam

from sdriving.agents.buffer import CentralizedPPOBuffer
from sdriving.agents.model import PPOLidarActorCritic
from sdriving.agents.utils import (
    count_vars,
    hvd_average_grad,
    hvd_scalar_statistics,
    trainable_parameters,
)
from sdriving.logging import EpochLogger, convert_json


class PPO_Distributed_Centralized_Critic:
    def __init__(
        self,
        env,
        env_params: dict,
        log_dir: str,
        ac_kwargs: dict = {},
        seed: int = 0,
        steps_per_epoch: int = 4000,
        epochs: int = 50,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_iters: int = 100,
        entropy_coeff: float = 1e-2,
        lam: float = 0.97,
        target_kl: float = 0.01,
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
        self.ac_params = {k: v for k, v in ac_kwargs.items()}
        self.ac_params.update(
            {
                "observation_space": self.env.observation_space,
                "action_space": self.env.action_space,
                "nagents": self.env.nagents,
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
        self.ac = PPOLidarActorCritic(
            self.env.observation_space,
            self.env.action_space,
            nagents=self.nagents,
            centralized=True,
            **ac_kwargs,
        )

        self.device = device

        self.pi_lr = pi_lr
        self.vf_lr = vf_lr

        self.load_path = load_path
        if load_path is not None:
            self.load_model(load_path)
        else:
            self.pi_optimizer = Adam(
                trainable_parameters(self.ac.pi), lr=self.pi_lr, eps=1e-8
            )
            self.vf_optimizer = Adam(
                trainable_parameters(self.ac.v), lr=self.vf_lr, eps=1e-8
            )

        # Sync params across processes
        hvd.broadcast_parameters(self.ac.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.pi_optimizer, root_rank=0)
        hvd.broadcast_optimizer_state(self.vf_optimizer, root_rank=0)
        self.ac = self.ac.to(device)
        self.move_optimizer_to_device(self.pi_optimizer)
        self.move_optimizer_to_device(self.vf_optimizer)

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

            wandb.watch(self.ac.pi, log="all")
            wandb.watch(self.ac.v, log="all")

        # Count variables
        var_counts = tuple(
            count_vars(module) for module in [self.ac.pi, self.ac.v]
        )
        self.logger.log(
            "\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts,
            color="green",
        )

        # Set up experience buffer
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = int(steps_per_epoch / hvd.size())
        self.buf = CentralizedPPOBuffer(
            self.env.observation_space[0].shape,
            self.env.observation_space[1].shape,
            self.env.action_space.shape,
            self.local_steps_per_epoch,
            gamma,
            lam,
            self.env.nagents,
            device=self.device,
        )

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.epochs = epochs
        self.save_freq = save_freq

    def compute_loss(self, data: dict, idxs):
        self.device
        clip_ratio = self.clip_ratio

        obs, lidar, act, adv, logp_old, vest, ret, mask = [
            data[k][:, idxs]
            for k in [
                "obs",
                "lidar",
                "act",
                "adv",
                "logp",
                "vest",
                "ret",
                "mask",
            ]
        ]

        # Value Function Loss
        value_est = self.ac.v((obs, lidar), mask).view(
            obs.size(0), obs.size(1)
        )
        value_est_clipped = vest + (value_est - vest).clamp(
            -clip_ratio, clip_ratio
        )
        value_losses = (value_est - ret).pow(2)
        value_losses_clipped = (value_est_clipped - ret).pow(2)

        value_loss = (
            0.5 * (torch.max(value_losses, value_losses_clipped) * mask).mean()
        )

        # Policy loss
        pi, _, logp = self.ac.pi((obs, lidar), act)
        ratio = torch.exp(logp - logp_old)  # N x B
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = torch.min(ratio * adv, clip_adv)
        loss_pi = -(loss_pi * mask).sum() / mask.sum()

        # Entropy Loss
        ent = pi.entropy().mean()

        # TODO: Search for a good set of coeffs
        loss = loss_pi - ent * self.entropy_coeff + value_loss
        # self.entropy_coeff -= self.entropy_coeff_decay

        # Logging Utilities
        approx_kl = (logp_old - logp).mean().detach().cpu()
        info = dict(
            kl=approx_kl,
            ent=ent.item(),
            value_est=value_est.mean().item(),
            pi_loss=loss_pi.item(),
            vf_loss=value_loss.item(),
        )

        return loss, info

    def update(self):
        data = self.buf.get()
        local_steps_per_epoch = self.local_steps_per_epoch
        train_iters = self.train_iters
        batch_size = local_steps_per_epoch // train_iters
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(
                torch.arange(0, data["obs"].size(1))
            ),
            batch_size=batch_size,
            drop_last=False,
        )

        early_stop = False
        for i, idxs in enumerate(sampler):  # range(train_iters):
            self.pi_optimizer.zero_grad()
            self.vf_optimizer.zero_grad()

            loss, info = self.compute_loss(data, idxs)

            kl = hvd.allreduce(info["kl"], op=hvd.Average)
            loss.backward()
            hvd_average_grad(self.ac, self.device)
            self.vf_optimizer.step()
            if kl > 1.5 * self.target_kl and not early_stop:
                self.logger.log(
                    f"Early stopping at step {i} due to reaching max kl.",
                    color="red",
                )
                early_stop = True
                self.logger.store(StopIter=i)
            if not early_stop:
                self.pi_optimizer.step()
        if not early_stop:
            self.logger.store(StopIter=i)

        self.env.sync()

        # Log changes from update
        ent, pi_l_old, v_l_old, v_est = (
            info["ent"],
            info["pi_loss"],
            info["vf_loss"],
            info["value_est"],
        )
        self.logger.store(
            LossActor=pi_l_old,
            LossCritic=v_l_old,
            KL=kl,
            Entropy=ent,
            ValueEstimate=v_est,
        )

    def move_optimizer_to_device(self, opt):
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def save_model(self, epoch: int, ckpt_extra: dict = {}):
        ckpt = {
            "actor": self.ac.pi.state_dict(),
            "critic": self.ac.v.state_dict(),
            "nagents": self.nagents,
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "ac_kwargs": self.ac_params,
            "entropy_coeff": self.entropy_coeff,
            "model": "centralized_critic",
        }
        ckpt.update(ckpt_extra)
        torch.save(
            ckpt,
            os.path.join(self.ckpt_dir, f"epoch_{epoch}_{time.time()}.pth"),
        )
        torch.save(ckpt, self.softlink)
        wandb.save(self.softlink)

    def load_model(self, load_path):
        ckpt = torch.load(load_path, map_location="cpu")
        self.ac.pi.load_state_dict(ckpt["actor"])
        self.pi_optimizer = Adam(
            trainable_parameters(self.ac.pi), lr=self.pi_lr, eps=1e-8
        )
        self.pi_optimizer.load_state_dict(ckpt["pi_optimizer"])
        if "entropy_coeff" in ckpt:
            self.entropy_coeff = ckpt["entropy_coeff"]
        if ckpt["nagents"] == self.nagents:
            self.ac.v.load_state_dict(ckpt["critic"])
            self.vf_optimizer = Adam(
                trainable_parameters(self.ac.v), lr=self.vf_lr, eps=1e-8
            )
            self.vf_optimizer.load_state_dict(ckpt["vf_optimizer"])
        else:
            self.vf_optimizer = Adam(
                trainable_parameters(self.ac.v), lr=self.vf_lr, eps=1e-8
            )
            self.logger.log("The agent was trained with a different nagents")
            if (
                "permutation_invariant" in self.ac_params
                and self.ac_params["permutation_invariant"]
            ):
                self.ac.v.load_state_dict(ckpt["critic"])
                self.vf_optimizer.load_state_dict(ckpt["vf_optimizer"])
                self.logger.log(
                    "Agent doesn't depend on nagents. So continuing finetuning"
                )

    def dump_tabular(self):
        self.logger.log_tabular("Epoch", average_only=True)
        self.logger.log_tabular("EpisodeReturn", with_min_and_max=True)
        self.logger.log_tabular("EpisodeLength", average_only=True)
        self.logger.log_tabular("Entropy", average_only=True)
        self.logger.log_tabular("KL", average_only=True)
        self.logger.log_tabular("StopIter", average_only=True)
        self.logger.log_tabular("ValueEstimate", average_only=True)
        self.logger.log_tabular("LossActor", average_only=True)
        self.logger.log_tabular("LossCritic", average_only=True)
        self.logger.log_tabular("EpisodeRunTime", average_only=True)
        self.logger.log_tabular("PPOUpdateTime", average_only=True)
        self.logger.dump_tabular()

    def train(self):
        # Prepare for interaction with environment
        for epoch in range(self.epochs):

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

        (o, a_ids), ep_ret, ep_len = env.reset(), 0, 0
        prev_done = torch.zeros(env.nagents, 1, device=self.device).bool()
        for t in range(self.local_steps_per_epoch):
            obs, lidar = o
            actions, val_f, log_probs = self.ac.step(
                [t.to(self.device) for t in o]
            )
            (next_o, _a_ids), r, d, info = self.env.step(actions)

            ep_ret += r.sum()
            ep_len += 1

            done = d.all()

            for i, name in enumerate(a_ids):
                if prev_done[i]:
                    continue
                b = int(name.rsplit("_", 1)[-1])
                self.buf.store(
                    b,
                    obs[i],
                    lidar[i],
                    actions[i],
                    r[i],
                    val_f[i],
                    log_probs[i],
                )

            o, a_ids = next_o, _a_ids
            prev_done = d

            timeout = info["timeout"] if "timeout" in info else done
            terminal = done or timeout
            epoch_ended = t == self.local_steps_per_epoch - 1

            if terminal or epoch_ended:
                v = torch.zeros(env.actual_nagents, device=self.device)
                if epoch_ended and not terminal:
                    _, _v, _ = self.ac.step([t.to(self.device) for t in o])
                    for i, a_id in enumerate(a_ids):
                        v[int(a_id.rsplit("_", 1)[-1])] = _v[i]

                self.buf.finish_path(v)

                if terminal:
                    ep_ret = ep_ret / self.env.actual_nagents
                    self.env.register_reward(ep_ret.cpu())
                    self.logger.store(
                        EpisodeReturn=ep_ret,
                        EpisodeLength=ep_len,
                    )
                (o, a_ids), ep_ret, ep_len = env.reset(), 0, 0
                prev_done = torch.zeros(
                    env.nagents, 1, device=self.device
                ).bool()
