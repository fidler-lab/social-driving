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
from sdriving.agents.alt_opt.runner import episode_runner
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


class PPO_Alternating_Optimization:
    def __init__(
        self,
        env,
        env_params: dict,
        log_dir: str,
        ac_kwargs: dict = {},
        actor_kwargs: dict = {},
        seed: int = 0,
        extra_iters: int = 5,
        steps_per_epoch_controller: int = 32000,
        steps_per_epoch_spline: int = 1600,
        epochs: int = 50,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        spline_lr: float = 3e-4,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_spline_iters: int = 1,
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        entropy_coeff: float = 1e-2,
        lam: float = 0.97,
        target_kl: float = 0.01,
        logger_kwargs: dict = {},
        save_freq: int = 10,
        load_path=None,
        wandb_id: Optional[str] = None,
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

        self.ac_params = {k: v for k, v in ac_kwargs.items()}
        self.ac_params.update(
            {
                "observation_space": self.env.controller_observation_space,
                "action_space": self.env.controller_action_space,
                "nagents": self.env.nagents,
            }
        )
        self.actor_params = {k: v for k, v in actor_kwargs.items()}
        self.actor_params.update(
            {
                "obs_dim": self.env.spline_observation_space.shape[0],
                "act_space": self.env.spline_action_space,
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

        self.nagents = self.env.nagents
        if isinstance(self.env.spline_action_space, gym.spaces.Discrete):
            self.actor = PPOWaypointCategoricalActor(**self.actor_params)
        elif isinstance(self.env.spline_action_space, gym.spaces.Box):
            self.actor = PPOWaypointGaussianActor(**self.actor_params)
        self.ac = PPOLidarActorCritic(
            self.env.controller_observation_space,
            self.env.controller_action_space,
            nagents=self.nagents,
            centralized=True,
            **ac_kwargs,
        )

        self.device = device

        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.spline_lr = spline_lr

        self.load_path = load_path
        if load_path is not None:
            self.load_model(load_path)
        else:
            self.spline_optimizer = Adam(
                trainable_parameters(self.actor), lr=self.spline_lr, eps=1e-8
            )
            self.pi_optimizer = Adam(
                trainable_parameters(self.ac.pi), lr=self.pi_lr, eps=1e-8
            )
            self.vf_optimizer = Adam(
                trainable_parameters(self.ac.v), lr=self.vf_lr, eps=1e-8
            )

        # Sync params across processes
        sync_params(self.actor)
        self.actor = self.actor.to(device)
        sync_params(self.ac)
        self.ac = self.ac.to(device)

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
            wandb.watch(self.ac.pi, log="all")
            wandb.watch(self.ac.v, log="all")

        # Count variables
        var_counts = tuple(
            count_vars(module)
            for module in [self.ac.pi, self.ac.v, self.actor]
        )
        self.logger.log(
            "\nNumber of parameters: \t pi: %d, \t v: %d \t spline: %d\n"
            % var_counts
        )

        # Set up experience buffer
        self.steps_per_epoch_controller = steps_per_epoch_controller
        self.local_steps_per_epoch_controller = int(
            steps_per_epoch_controller / num_procs()
        )
        self.steps_per_epoch_spline = steps_per_epoch_spline
        self.local_steps_per_epoch_spline = int(
            steps_per_epoch_spline / num_procs()
        )

        self.buf_spline = OneStepPPOBuffer(
            self.env.spline_observation_space.shape,
            self.env.spline_action_space.shape,
            self.local_steps_per_epoch_spline,
            self.env.nagents,
        )
        if (
            "permutation_invariant" in self.ac_params
            and self.ac_params["permutation_invariant"]
        ):
            self.buf_controller = CentralizedPPOBuffer(  # VariableNagents(
                self.env.controller_observation_space[0].shape,
                self.env.controller_observation_space[1].shape,
                self.env.controller_action_space.shape,
                self.local_steps_per_epoch_controller,
                gamma,
                lam,
                self.env.nagents,
            )
        else:
            self.buf_controller = CentralizedPPOBuffer(
                self.env._controllerobservation_space[0].shape,
                self.env._controllerobservation_space[1].shape,
                self.env._controlleraction_space.shape,
                self.local_steps_per_epoch,
                gamma,
                lam,
                self.env.nagents,
            )

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_spline_iters = train_spline_iters
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.epochs = epochs
        self.save_freq = save_freq
        self.extra_iters = extra_iters

    def compute_controller_loss(self, data, idx):
        device = self.device
        clip_ratio = self.clip_ratio

        # Random subset sampling
        data = {
            a_id: {key: data[a_id][key][idx] for key in data[a_id].keys()}
            for a_id in data.keys()
        }
        for a_id in data.keys():
            adv_mean, adv_std = (
                data[a_id]["adv"].mean(),
                data[a_id]["adv"].std(),
            )
            data[a_id]["adv"] = (data[a_id]["adv"] - adv_mean) / (
                adv_std + 1e-7
            )

        # Convert the data to the required format for actor and critic
        data_vf = data
        data_pi = {}
        [
            [self.make_entry(data_pi, k, d[k]) for k in d.keys()]
            for d in data.values()
        ]

        obs, lidar, act, adv, logp_old = [
            data_pi[k].to(device)
            for k in ["obs", "lidar", "act", "adv", "logp"]
        ]

        obs_list = []
        vest_old = sum(d["vest"] for d in data_vf.values()).to(device) / len(
            data_vf.keys()
        )
        ret = sum(d["ret"] for d in data_vf.values()).to(device) / len(
            data_vf.keys()
        )
        for data_val in data_vf.values():
            if isinstance(self.ac, PPOLidarActorCritic):
                obs_list.append(
                    (data_val["obs"].to(device), data_val["lidar"].to(device))
                )
            elif isinstance(self.ac, PPOWaypointActorCritic):
                obs_list.append(data_val["obs"].to(device))

        # Value Function Loss
        value_est = self.ac.v(obs_list)
        value_est_clipped = vest_old + (value_est - vest_old).clamp(
            -clip_ratio, clip_ratio
        )
        value_losses = (value_est - ret).pow(2)
        value_losses_clipped = (value_est_clipped - ret).pow(2)

        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        # Policy loss
        if isinstance(self.ac, PPOLidarActorCritic):
            pi, logp = self.ac.pi((obs, lidar), act)
        elif isinstance(self.ac, PPOWaypointActorCritic):
            pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp((logp - logp_old).clamp(-20.0, 2.0))
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Entropy Loss
        ent = pi.entropy().mean()

        # TODO: Search for a good set of coeffs
        loss = loss_pi - ent * self.entropy_coeff + value_loss
        self.entropy_coeff -= self.entropy_coeff_decay

        # Logging Utilities
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        info = dict(
            kl=approx_kl,
            ent=ent.item(),
            cf=clipfrac,
            value_est=value_est.mean().item(),
            pi_loss=loss_pi.item(),
            vf_loss=value_loss.item(),
        )

        return loss, info

    def compute_spline_loss(self, data: dict, idx):
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
            kl=approx_kl, ent=ent.item(), cf=clipfrac, pi_loss=loss_pi.item(),
        )

        return loss, info

    def spline_update(self):
        data = self.buf_spline.get()

        train_iters = self.train_spline_iters
        size = int(mpi_op(data["obs"].size(0), MPI.MIN))
        batch_size = size // train_iters

        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(range(size)),
            batch_size,
            drop_last=True,
        )

        with torch.no_grad():
            _, info = self.compute_spline_loss(data, range(size))
            pi_l_old = info["pi_loss"]

        for i, idx in enumerate(sampler):
            self.spline_optimizer.zero_grad()

            loss, info = self.compute_spline_loss(data, idx)

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

            self.spline_optimizer.step()

        self.logger.store(StopIterSpline=i)

        # Log changes from update
        kl, ent, cf = info["kl"], info["ent"], info["cf"]
        self.logger.store(
            LossPiSpline=pi_l_old,
            KLSpline=kl,
            EntropySpline=ent,
            ClipFracSpline=cf,
            DeltaLossPiSpline=(info["pi_loss"] - pi_l_old),
        )
        if proc_id() == 0:
            wandb.log(
                {
                    "Loss Actor (Spline)": pi_l_old,
                    "KL Divergence (Spline)": kl,
                    "Entropy (Spline)": ent,
                    "Clip Factor (Spline)": cf,
                }
            )

    @staticmethod
    def make_entry(d: dict, k: str, val: torch.Tensor):
        if k not in d:
            d[k] = val
        else:
            d[k] = torch.cat([d[k], val], dim=0)

    def controller_update(self):
        data = self.buf_controller.get()
        local_steps_per_epoch = self.local_steps_per_epoch_controller

        train_iters = max(self.train_pi_iters, self.train_v_iters)
        batch_size = local_steps_per_epoch // train_iters

        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(range(local_steps_per_epoch)),
            batch_size,
            drop_last=True,
        )

        with torch.no_grad():
            _, info = self.compute_controller_loss(
                data, range(local_steps_per_epoch)
            )
            pi_l_old = info["pi_loss"]
            v_est = info["value_est"]
            v_l_old = info["vf_loss"]

        for i, idx in enumerate(sampler):
            self.pi_optimizer.zero_grad()
            self.vf_optimizer.zero_grad()

            loss, info = self.compute_controller_loss(data, idx)

            kl = mpi_avg(info["kl"])
            if kl > 1.5 * self.target_kl:
                self.logger.log(
                    f"Early stopping at step {i} due to reaching max kl."
                )
                break
            loss.backward()

            self.ac.pi = self.ac.pi.cpu()
            mpi_avg_grads(self.ac.pi)
            self.ac.pi = self.ac.pi.to(self.device)

            self.ac.v = self.ac.v.cpu()
            mpi_avg_grads(self.ac.v)
            self.ac.v = self.ac.v.to(self.device)

            self.pi_optimizer.step()
            self.vf_optimizer.step()

        self.logger.store(StopIterController=i)

        # Log changes from update
        kl, ent, cf = info["kl"], info["ent"], info["cf"]
        self.logger.store(
            LossPiController=pi_l_old,
            LossVController=v_l_old,
            KLController=kl,
            EntropyController=ent,
            ClipFracController=cf,
            DeltaLossPiController=(info["pi_loss"] - pi_l_old),
            DeltaLossVController=(info["vf_loss"] - v_l_old),
            ValueEstimateController=v_est,
        )
        if proc_id() == 0:
            wandb.log(
                {
                    "Loss Actor (Controller)": pi_l_old,
                    "Loss Value Function (Controller)": v_l_old,
                    "KL Divergence (Controller)": kl,
                    "Entropy (Controller)": ent,
                    "Clip Factor (Controller)": cf,
                    "Value Estimate (Controller)": v_est,
                }
            )

    def save_model(self, epoch: int, ckpt_extra: dict = {}):
        ckpt = {
            "spline": self.actor.state_dict(),
            "actor": self.ac.pi.state_dict(),
            "critic": self.ac.v.state_dict(),
            "nagents": self.nagents,
            "spline_optimizer": self.spline_optimizer.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "actor_kwargs": self.actor_params,
            "ac_kwargs": self.ac_params,
            "model": "centralized_critic",
            "type": "alt_opt",
        }
        ckpt.update(ckpt_extra)
        torch.save(ckpt, self.softlink)
        wandb.save(self.softlink)

    def load_model(self, load_path):
        ckpt = torch.load(load_path, map_location="cpu")
        self.actor.load_state_dict(ckpt["spline"])
        self.spline_optimizer = Adam(
            trainable_parameters(self.actor), lr=self.spline_lr, eps=1e-8
        )
        self.spline_optimizer.load_state_dict(ckpt["spline_optimizer"])
        for state in self.spline_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        self.ac.pi.load_state_dict(ckpt["actor"])
        self.pi_optimizer = Adam(
            trainable_parameters(self.ac.pi), lr=self.pi_lr, eps=1e-8
        )
        self.pi_optimizer.load_state_dict(ckpt["pi_optimizer"])
        for state in self.pi_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
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
            self.logger.log(
                "The agent was trained with a different nagents", color="red",
            )
            if (
                "permutation_invariant" in self.ac_params
                and self.ac_params["permutation_invariant"]
            ):
                self.ac.v.load_state_dict(ckpt["critic"])
                self.vf_optimizer.load_state_dict(ckpt["vf_optimizer"])
                self.logger.log(
                    "Agent doesn't depend on nagents. So continuing finetuning",
                    color="green",
                )
        for state in self.vf_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def dump_logs(self, epoch, start_time, spline=False):
        self.logger.log_tabular("Epoch", epoch)
        if spline:
            self.logger.log_tabular("EpRetSpline", with_min_and_max=True)
            self.logger.log_tabular("LossPiSpline", average_only=True)
            self.logger.log_tabular("DeltaLossPiSpline", average_only=True)
            self.logger.log_tabular("EntropySpline", average_only=True)
            self.logger.log_tabular("KLSpline", average_only=True)
            self.logger.log_tabular("ClipFracSpline", average_only=True)
            self.logger.log_tabular("StopIterSpline", average_only=True)
        self.logger.log_tabular("EpRetController", with_min_and_max=True)
        self.logger.log_tabular("EpLenController", average_only=True)
        self.logger.log_tabular("VValsController", average_only=True)
        self.logger.log_tabular("LossPiController", average_only=True)
        self.logger.log_tabular("LossVController", average_only=True)
        self.logger.log_tabular("DeltaLossPiController", average_only=True)
        self.logger.log_tabular("DeltaLossVController", average_only=True)
        self.logger.log_tabular("EntropyController", average_only=True)
        self.logger.log_tabular("KLController", average_only=True)
        self.logger.log_tabular("ClipFracController", average_only=True)
        self.logger.log_tabular("StopIterController", average_only=True)
        self.logger.log_tabular("Time", time.time() - start_time)
        self.logger.dump_tabular()

    def train(self):
        # Prepare for interaction with environment
        start_time = time.time()

        for epoch in range(self.epochs):
            episode_runner(
                self.local_steps_per_epoch_controller,
                self.device,
                self.buf_controller,
                self.env,
                self.actor,
                self.ac,
                self.logger,
                spline=False,
            )

            self.controller_update()

            if epoch % self.extra_iters == 0:
                episode_runner(
                    self.local_steps_per_epoch_spline,
                    self.device,
                    self.buf_spline,
                    self.env,
                    self.actor,
                    self.ac,
                    self.logger,
                    spline=True,
                )

                self.spline_update()

            if (
                (epoch % self.save_freq == 0) or (epoch == self.epochs - 1)
            ) and proc_id() == 0:
                self.save_model(epoch)

            # Log info about epoch
            self.dump_logs(
                epoch, start_time, spline=epoch % self.extra_iters == 0
            )
