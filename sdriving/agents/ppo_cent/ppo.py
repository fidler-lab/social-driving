import os
import time
import warnings
from typing import Optional

import gym
import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter

from sdriving.agents.ppo_cent.model import ActorCritic
from sdriving.agents.ppo_cent.utils import (
    PPOBuffer,
    count_vars,
    mpi_avg_grads,
    trainable_parameters,
)
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

import wandb


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
        lam: float = 0.97,
        target_kl: float = 0.01,
        logger_kwargs: dict = {},
        save_freq: int = 10,
        load_path=None,
        render_train: bool = False,
        tboard: bool = True,
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
            **ac_kwargs,
        )
        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            self.ac.pi.load_state_dict(ckpt["actor"])
            self.pi_optimizer = Adam(trainable_parameters(self.ac.pi), lr=pi_lr)
            self.pi_optimizer.load_state_dict(ckpt["pi_optimizer"])
            for state in self.pi_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            if ckpt["nagents"] == self.nagents:
                self.ac.v.load_state_dict(ckpt["critic"])
                self.vf_optimizer = Adam(trainable_parameters(self.ac.v), lr=vf_lr)
                self.vf_optimizer.load_state_dict(ckpt["vf_optimizer"])
            else:
                self.vf_optimizer = Adam(trainable_parameters(self.ac.v), lr=vf_lr)
                self.logger.log(
                    "The agent was trained with a different nagents",
                    color="red",
                )
            for state in self.vf_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            self.pi_optimizer = Adam(trainable_parameters(self.ac.pi), lr=pi_lr)
            self.vf_optimizer = Adam(trainable_parameters(self.ac.v), lr=vf_lr)

        # Sync params across processes
        sync_params(self.ac)
        self.ac = self.ac.to(device)
        
        if proc_id() == 0:
            eid = log_dir.split('/')[-2] if load_path is None else load_path.split('/')[-4]
            wandb.init(
                name=eid,
                id=eid,
                project="Social Driving",
                resume=load_path is not None,
                allow_val_change=True
            )
            wandb.watch_called = False

            if "self" in hparams:
                del hparams["self"]
            wandb.config.update(hparams, allow_val_change=True)

            wandb.watch(self.ac.pi.lidar_features, log="all")
            wandb.watch(self.ac.pi.logits_net, log="all")
            wandb.watch(self.ac.v, log="all")

        self.device = device

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
        self.buf = PPOBuffer(
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

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        device = self.device
        clip_ratio = self.clip_ratio

        obs, lidar, act, adv, logp_old = (
            data["obs"].to(device),
            data["lidar"].to(device),
            data["act"].to(device),
            data["adv"].to(device),
            data["logp"].to(device),
        )

        # Policy loss
        pi, logp = self.ac.pi((obs, lidar), act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        device = self.device

        obs_list = []
        ret = sum(d["ret"] for d in data.values()).to(device) / len(
            data.keys()
        )
        for data_val in data.values():
            obs_list.append(
                (data_val["obs"].to(device), data_val["lidar"].to(device))
            )

        value_est = self.ac.v(obs_list)

        return (
            ((value_est - ret) ** 2).mean(),
            dict(value_est=value_est.mean().item()),
        )

    @staticmethod
    def make_entry(d: dict, k: str, val: torch.Tensor):
        if k not in d:
            d[k] = val
        else:
            d[k] = torch.cat([d[k], val], dim=0)

    def update(self, epoch, t):
        data = self.buf.get()
        local_steps_per_epoch = self.local_steps_per_epoch

        data_pi = {}
        for d in data.values():
            self.make_entry(data_pi, "obs", d["obs"])
            self.make_entry(data_pi, "lidar", d["lidar"])
            self.make_entry(data_pi, "act", d["act"])
            self.make_entry(data_pi, "adv", d["adv"])
            self.make_entry(data_pi, "logp", d["logp"])

        with torch.no_grad():
            pi_l_old, pi_info_old = self.compute_loss_pi(data_pi)
            pi_l_old = pi_l_old.item()
            v_l_old, v_est = self.compute_loss_v(data)
            v_l_old = v_l_old.item()
            v_est = v_est["value_est"]

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data_pi)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * self.target_kl:
                self.logger.log(
                    "Early stopping at step %d due to reaching max kl." % i
                )
                break
            loss_pi.backward()
            self.ac.pi = self.ac.pi.cpu()
            # TODO: Use NCCL to do this without GPU to CPU transfer
            mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
            self.ac.pi = self.ac.pi.to(self.device)
            self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v, _ = self.compute_loss_v(data)
            loss_v.backward()
            self.ac.v = self.ac.v.cpu()
            # TODO: Use NCCL
            mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.ac.v = self.ac.v.to(self.device)
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        self.logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
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
                    "Value Estimate": v_est
                }
            )

    def train(self):
        env = self.env
        ac = self.ac

        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0

        for epoch in range(self.epochs):
            for t in range(self.local_steps_per_epoch):
                a = {}
                v = {}
                logp = {}
                o_list = []
                for key, obs in o.items():
                    obs = tuple([t.detach().to(self.device) for t in obs])
                    o[key] = obs
                    o_list.append(obs)

                actions, val_f, log_probs = ac.step(o_list)
                for i, key in enumerate(o.keys()):
                    a[key] = actions[i]
                    v[key] = val_f
                    logp[key] = log_probs[i]
                next_o, r, d, info = env.step(a)
                rlist = [
                    torch.as_tensor(rwd).detach().cpu()
                    for _, rwd in r.items()
                ]
                ret = sum(rlist)
                rlen = len(rlist)
                ep_ret += ret / rlen
                ep_len += 1

                # save and log
                done = d["__all__"]

                # Store experience to replay buffer
                for key, obs in o.items():
                    self.buf.store(
                        key,
                        obs[0].cpu(),
                        obs[1].cpu(),
                        a[key].cpu(),
                        torch.as_tensor(r[key]).detach().cpu(),
                        v[key].cpu(),
                        logp[key].cpu(),
                    )
                    self.logger.store(VVals=v[key])

                # Update obs (critical!)
                o = next_o

                timeout = info["timeout"] if "timeout" in info else done
                terminal = done or timeout
                epoch_ended = t == self.local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    # if trajectory didn't reach terminal state,
                    # bootstrap value target
                    if timeout or epoch_ended:
                        o_list = []
                        for _, obs in o.items():
                            o_list.append(
                                tuple([t.to(self.device) for t in obs])
                            )
                        _, v, _ = ac.step(o_list)
                        v = v.cpu()
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    # only save EpRet / EpLen if trajectory finished
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if terminal:
                        if proc_id() == 0:
                            wandb.log({
                                "Episode Return (Train)": ep_ret,
                                "Episode Length (Train)": ep_len
                            })
                    o, ep_ret, ep_len = env.reset(), 0, 0

            if (
                (epoch % self.save_freq == 0) or (epoch == self.epochs - 1)
            ) and proc_id() == 0:
                ckpt = {
                    "actor": self.ac.pi.state_dict(),
                    "critic": self.ac.v.state_dict(),
                    "nagents": self.nagents,
                    "pi_optimizer": self.pi_optimizer.state_dict(),
                    "vf_optimizer": self.vf_optimizer.state_dict(),
                    "ac_kwargs": self.ac_params,
                    "model": "centralized_critic",
                }
                filename = os.path.join(self.ckpt_dir, f"ckpt_{epoch}.pth")
                torch.save(ckpt, filename)
                torch.save(ckpt, self.softlink)
                wandb.save(self.softlink)

            self.update(epoch, t)

            # Log info about epoch
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
