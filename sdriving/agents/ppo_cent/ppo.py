import os
import time
import warnings
from typing import Optional

import gym
import numpy as np
import torch
import wandb
from sdriving.agents.buffer import (
    CentralizedPPOBuffer,
    CentralizedPPOBufferVariableNagents
)
from sdriving.agents.model import PPOLidarActorCritic as ActorCritic
from sdriving.agents.model import IterativeWayPointPredictor
from sdriving.agents.utils import (
    count_vars,
    mpi_avg_grads,
    trainable_parameters,
)
from sdriving.agents.ppo_cent.runner import episode_runner
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from spinup.utils.mpi_tools import (
    mpi_avg,
    mpi_fork,
    mpi_statistics_scalar,
    num_procs,
    proc_id,
)
from spinup.utils.serialization_utils import convert_json
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter


class PPO_Centralized_Critic:
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
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        entropy_coeff: float = 1e-2,
        lam: float = 0.97,
        target_kl: float = 0.01,
        logger_kwargs: dict = {},
        save_freq: int = 10,
        load_path=None,
        render_train: bool = False,
        tboard: bool = True,
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
        self.ac_params_file = os.path.join(log_dir, "ac_params.json")

        self.logger.save_config(locals())

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

        if torch.cuda.is_available():
            # From emperical results, 8 tasks can use a single gpu
            device_id = proc_id() // 8
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")

        torch.save(self.ac_params, self.ac_params_file)

        if os.path.isfile(self.softlink):
            self.logger.log("Restarting from latest checkpoint", color="red")
            load_path = self.softlink

        # if tboard and proc_id() == 0:
        #     self.writer = SummaryWriter(log_dir)

        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        if render_train:
            self.logger.log(
                "Rendering the training is not implemented", color="red"
            )

        self.nagents = self.env.nagents
        self.ac = ActorCritic(
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

            wandb.watch(self.ac.pi, log="all")
            wandb.watch(self.ac.v, log="all")

        # Count variables
        var_counts = tuple(
            count_vars(module) for module in [self.ac.pi, self.ac.v]
        )
        self.logger.log(
            "\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts
        )

        # Set up experience buffer
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        if "permutation_invariant" in self.ac_params and self.ac_params["permutation_invariant"]:
            self.buf = CentralizedPPOBuffer(  # VariableNagents(
                self.env.observation_space[0].shape,
                self.env.observation_space[1].shape,
                self.env.action_space.shape,
                self.local_steps_per_epoch,
                gamma,
                lam,
                self.env.nagents
            )
        else:
            self.buf = CentralizedPPOBuffer(
                self.env.observation_space[0].shape,
                self.env.observation_space[1].shape,
                self.env.action_space.shape,
                self.local_steps_per_epoch,
                gamma,
                lam,
                self.env.nagents,
            )

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.epochs = epochs
        self.save_freq = save_freq
        self.tboard = tboard

    def compute_loss(self, data, idx):
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
            obs_list.append(
                (data_val["obs"].to(device), data_val["lidar"].to(device))
            )

        # Value Function Loss
        value_est = self.ac.v(obs_list)
        value_est_clipped = vest_old + (value_est - vest_old).clamp(
            -clip_ratio, clip_ratio
        )
        value_losses = (value_est - ret).pow(2)
        value_losses_clipped = (value_est_clipped - ret).pow(2)

        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        # Policy loss
        pi, logp = self.ac.pi((obs, lidar), act)
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

    @staticmethod
    def make_entry(d: dict, k: str, val: torch.Tensor):
        if k not in d:
            d[k] = val
        else:
            d[k] = torch.cat([d[k], val], dim=0)

    def update(self):
        data = self.buf.get()
        local_steps_per_epoch = self.local_steps_per_epoch

        train_iters = max(self.train_pi_iters, self.train_v_iters)
        batch_size = local_steps_per_epoch // train_iters

        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(range(local_steps_per_epoch)),
            batch_size,
            drop_last=True,
        )

        with torch.no_grad():
            _, info = self.compute_loss(data, range(local_steps_per_epoch))
            pi_l_old = info["pi_loss"]
            v_est = info["value_est"]
            v_l_old = info["vf_loss"]

        for i, idx in enumerate(sampler):
            self.pi_optimizer.zero_grad()
            self.vf_optimizer.zero_grad()

            loss, info = self.compute_loss(data, idx)

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

        self.logger.store(StopIter=i)

        # Log changes from update
        kl, ent, cf = info["kl"], info["ent"], info["cf"]
        self.logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(info["pi_loss"] - pi_l_old),
            DeltaLossV=(info["vf_loss"] - v_l_old),
            ValueEstimate=v_est,
        )
        if proc_id() == 0:
            wandb.log(
                {
                    "Loss Actor": pi_l_old,
                    "Loss Value Function": v_l_old,
                    "KL Divergence": kl,
                    "Entropy": ent,
                    "Clip Factor": cf,
                    "Value Estimate": v_est,
                }
            )

    def save_model(self, epoch: int, ckpt_extra: dict = {}):
        ckpt = {
            "actor": self.ac.pi.state_dict(),
            "critic": self.ac.v.state_dict(),
            "nagents": self.nagents,
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "ac_kwargs": self.ac_params,
            "model": "centralized_critic",
        }
        ckpt.update(ckpt_extra)
        # filename = os.path.join(self.ckpt_dir, f"ckpt_{epoch}.pth")
        # torch.save(ckpt, filename)
        torch.save(ckpt, self.softlink)
        wandb.save(self.softlink)

    def load_model(self, load_path):
        ckpt = torch.load(load_path, map_location="cpu")
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
            if "permutation_invariant" in self.ac_params and self.ac_params["permutation_invariant"]:
                self.ac.v.load_state_dict(ckpt["critic"])
                self.vf_optimizer.load_state_dict(ckpt["vf_optimizer"])
                self.logger.log(
                    "Agent doesn't depend on nagents. So continuing finetuning", color="green",
                )
        for state in self.vf_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def dump_logs(self, epoch, start_time):
        self.logger.log_tabular("Epoch", epoch)
        self.logger.log_tabular("EpRet", with_min_and_max=True)
        self.logger.log_tabular("EpLen", average_only=True)
        self.logger.log_tabular("VVals", with_min_and_max=True)
        self.logger.log_tabular(
            "TotalEnvInteracts", (epoch + 1) * self.steps_per_epoch
        )
        self.logger.log_tabular("LossPi", average_only=True)
        self.logger.log_tabular("LossV", average_only=True)
        self.logger.log_tabular("DeltaLossPi", average_only=True)
        self.logger.log_tabular("DeltaLossV", average_only=True)
        self.logger.log_tabular("Entropy", average_only=True)
        self.logger.log_tabular("KL", average_only=True)
        self.logger.log_tabular("ClipFrac", average_only=True)
        self.logger.log_tabular("StopIter", average_only=True)
        self.logger.log_tabular("ValueEstimate", average_only=True)
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
                self.ac,
                self.logger,
            )

            if (
                (epoch % self.save_freq == 0) or (epoch == self.epochs - 1)
            ) and proc_id() == 0:
                self.save_model(epoch)

            self.update()

            # Log info about epoch
            self.dump_logs(epoch, start_time)


class PPO_Centralized_Critic_AltOpt(PPO_Centralized_Critic):
    def __init__(
        self,
        *args,
        hidden_dim_wpoint: int = 64,
        separate_goal_model: bool = False,
        spline_lr: float = 1e-3,
        spline_model_iters: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.spline_args = {
            "hdim": hidden_dim_wpoint,
            "max_length": self.env.max_length,
            "max_width": self.env.max_width,
            "separate_goal_model": separate_goal_model,
        }
        self.spline_model = IterativeWayPointPredictor(**self.spline_args)
        sync_params(self.spline_model)
        self.opt_spline = Adam(
            self.spline_model.parameters(), lr=spline_lr, eps=1e-8
        )
        self.spline_model_iters = spline_model_iters
        self.load_spline_model(self.load_path)

    def save_model(self, epoch: int, ckpt_extra: dict = {}):
        ckpt_extra["spline"] = self.spline_model.state_dict()
        ckpt_extra["opt_spline"] = self.opt_spline.state_dict()
        ckpt_extra["model"] = "centralized_critic_spline"
        ckpt_extra["spline_kwargs"] = self.spline_args
        super().save_model(epoch, ckpt_extra)

    def load_spline_model(self, load_path):
        if load_path is None:
            return
        ckpt = torch.load(load_path, map_location="cpu")
        self.spline_model.load_state_dict(ckpt["spline"])
        self.opt_spline.load_state_dict(ckpt["opt_spline"])

    def dump_logs(self, epoch, start_time):
        self.logger.log_tabular("SplinePathLoss", with_min_and_max=True)
        super().dump_logs(epoch, start_time)

    def update_spline_model(self):
        for i in range(self.spline_model_iters):
            o = self.env.reset()

            self.opt_spline.zero_grad()
            self.env.register_track(self.spline_model)

            is_done = False
            losses = {a_id: 0.0 for a_id in self.env.get_agent_ids_list()}

            while not is_done:
                o_list = []
                for k, obs in o.items():
                    obs = tuple([t.detach().to(self.device) for t in obs])
                    o[k] = obs
                    o_list.append(obs)

                actions = {
                    a_id: self.ac.act(
                        tuple([t.detach().to(self.device) for t in o[a_id]]),
                        deterministic=True,
                    )
                    for a_id in o.keys()
                }

                o, loss, is_done, _ = self.env.step(
                    actions, differentiable_objective=True
                )

                losses = {
                    a_id: losses[a_id] + loss[a_id]
                    for a_id in self.env.get_agent_ids_list()
                }

                is_done = is_done["__all__"]

            mean_loss = sum(losses.values()) / len(
                self.env.get_agent_ids_list()
            )
            mean_loss.backward()

            mpi_avg_grads(self.spline_model)

            self.opt_spline.step()

            if proc_id() == 0:
                wandb.log({"Spline Path Loss": mean_loss.item()})
            self.logger.store(SplinePathLoss=mean_loss.item())

    def train(self):
        # Prepare for interaction with environment
        start_time = time.time()

        for epoch in range(self.epochs):
            self.update_spline_model()

            episode_runner(
                self.local_steps_per_epoch,
                self.device,
                self.buf,
                self.env,
                self.ac,
                self.logger,
                self.spline_model,
            )

            if (
                (epoch % self.save_freq == 0) or (epoch == self.epochs - 1)
            ) and proc_id() == 0:
                self.save_model(epoch)

            self.update()

            # Log info about epoch
            self.dump_logs(epoch, start_time)
