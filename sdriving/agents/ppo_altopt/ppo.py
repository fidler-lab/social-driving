import os
import time
from typing import Optional

import gym
import horovod.torch as hvd
import numpy as np
import torch
import wandb
from torch.optim import Adam

from sdriving.agents.buffer import CentralizedPPOBuffer, OneStepPPOBuffer
from sdriving.agents.model import (
    PPOLidarActorCritic,
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


class PPO_Alternating_Optimization_Centralized_Critic:
    def __init__(
        self,
        env,
        env_params: dict,
        log_dir: str,
        actor_kwargs: dict = dict(),
        ac_kwargs: dict = dict(),
        number_episodes_per_spline_update: int = 4000,
        number_steps_per_controller_update: int = 4000,
        seed: int = 0,
        epochs: int = 50,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        spline_lr: float = 3e-4,
        train_iters: int = 20,
        entropy_coeff: float = 1e-2,
        lam: float = 0.97,
        target_kl: float = 0.01,
        load_path=None,
        wandb_id: Optional[str] = None,
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

        self.env = env(**env_params)
        self.ac_params = {k: v for k, v in ac_kwargs.items()}
        self.ac_params.update(
            {
                "observation_space": self.env.observation_space[1],
                "action_space": self.env.action_space[1],
                "nagents": self.env.nagents,
                "centralized": True,
            }
        )
        self.actor_params = {k: v for k, v in actor_kwargs.items()}
        self.actor_params.update(
            {
                "obs_dim": self.env.observation_space[0].shape[0],
                "act_space": self.env.action_space[0],
            }
        )

        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff / epochs

        # Horovod: limit # of CPU threads to be used per worker.
        torch.set_num_threads(1)

        if os.path.isfile(self.softlink):
            self.logger.log("Restarting from latest checkpoint", color="red")
            load_path = self.softlink

        # Random seed
        seed += 10000 * hvd.rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.nagents = self.env.nagents

        if isinstance(self.env.action_space[0], gym.spaces.Discrete):
            self.actor = PPOWaypointCategoricalActor(**self.actor_params)
        elif isinstance(self.env.action_space[0], gym.spaces.Box):
            self.actor = PPOWaypointGaussianActor(**self.actor_params)
        self.ac = PPOLidarActorCritic(**self.ac_params)

        self.device = device

        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.spline_lr = spline_lr

        self.load_path = load_path

        if load_path is not None:
            self.load_model(load_path)
        else:
            self.pi_optimizer = Adam(
                trainable_parameters(self.ac.pi), lr=self.pi_lr, eps=1e-8
            )
            self.spline_optimizer = Adam(
                trainable_parameters(self.actor), lr=self.spline_lr, eps=1e-8
            )
            self.vf_optimizer = Adam(
                trainable_parameters(self.ac.v), lr=self.vf_lr, eps=1e-8
            )

        # Sync params across processes
        hvd.broadcast_parameters(self.ac.state_dict(), root_rank=0)
        hvd.broadcast_parameters(self.ac.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.pi_optimizer, root_rank=0)
        hvd.broadcast_optimizer_state(self.vf_optimizer, root_rank=0)
        hvd.broadcast_optimizer_state(self.spline_optimizer, root_rank=0)
        self.ac = self.ac.to(device)
        self.actor = self.actor.to(device)
        self.move_optimizer_to_device(self.pi_optimizer)
        self.move_optimizer_to_device(self.vf_optimizer)
        self.move_optimizer_to_device(self.spline_optimizer)

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
            wandb.watch(self.actor, log="all")

        # Count variables
        var_counts = tuple(
            count_vars(module)
            for module in [self.ac.pi, self.ac.v, self.actor]
        )
        self.logger.log(
            "\nNumber of parameters: \t pi: %d, \t v: %d, \t spline: %d\n"
            % var_counts,
            color="green",
        )

        self.number_episodes_per_spline_update = (
            number_episodes_per_spline_update
        )
        self.number_steps_per_controller_update = (
            number_steps_per_controller_update
        )
        self.local_number_episodes = int(
            self.number_episodes_per_spline_update / hvd.size()
        )
        self.local_steps_per_epoch = int(
            self.number_steps_per_controller_update / hvd.size()
        )

        self.spline_buffer = OneStepPPOBuffer(
            self.env.observation_space[0].shape,
            self.env.action_space[0].shape,
            self.local_number_episodes,
            self.env.nagents,
            self.device,
        )
        self.controller_buffer = CentralizedPPOBuffer(
            self.env.observation_space[1][0].shape,
            self.env.observation_space[1][1].shape,
            self.env.action_space[1].shape,
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

    def compute_spline_loss(self, data):
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

    def compute_controller_loss(self, data):
        self.device
        clip_ratio = self.clip_ratio

        obs, lidar, act, adv, logp_old, vest, ret, mask = [
            data[k]
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

    def move_optimizer_to_device(self, opt):
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def update_spline(self):
        data = self.spline_buffer.get()
        train_iters = self.train_iters

        for i in range(train_iters):
            self.spline_optimizer.zero_grad()

            loss, info = self.compute_spline_loss(data)

            kl = hvd.allreduce(info["kl"], op=hvd.Average)
            loss.backward()
            hvd_average_grad(self.actor, self.device)
            if kl > 1.5 * self.target_kl:
                self.logger.log(
                    f"Early stopping at step {i} due to reaching max kl.",
                    color="red",
                )
                break
            self.spline_optimizer.step()
        self.logger.store(StopIterSpline=i)

        # Log changes from update
        ent, pi_l_old = info["ent"], info["pi_loss"]
        self.logger.store(
            LossActorSpline=pi_l_old,
            KLSpline=kl,
            EntropySpline=ent,
        )

    def update_controller(self):
        data = self.controller_buffer.get()
        train_iters = self.train_iters

        for i in range(train_iters):
            self.pi_optimizer.zero_grad()
            self.vf_optimizer.zero_grad()

            loss, info = self.compute_controller_loss(data)

            kl = hvd.allreduce(info["kl"], op=hvd.Average)
            loss.backward()
            hvd_average_grad(self.ac, self.device)
            self.vf_optimizer.step()
            if kl > 1.5 * self.target_kl:
                self.logger.log(
                    f"Early stopping at step {i} due to reaching max kl.",
                    color="red",
                )
                break
            self.pi_optimizer.step()
        self.logger.store(StopIterController=i)

        # Log changes from update
        ent, pi_l_old, v_l_old, v_est = (
            info["ent"],
            info["pi_loss"],
            info["vf_loss"],
            info["value_est"],
        )
        self.logger.store(
            LossActorController=pi_l_old,
            LossCriticController=v_l_old,
            KLController=kl,
            EntropyController=ent,
            ValueEstimateController=v_est,
        )

    def save_model(self, epoch: int, ckpt_extra: dict = dict()):
        ckpt = {
            "controller_actor": self.ac.pi.state_dict(),
            "controller_critic": self.ac.v.state_dict(),
            "spline_actor": self.actor.state_dict(),
            "nagents": self.nagents,
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "spline_optimizer": self.spline_optimizer.state_dict(),
            "ac_kwargs": self.ac_params,
            "actor_kwargs": self.actor_params,
            "model": "centralized_critic",
            "type": "bilevel_model",
        }
        ckpt.update(ckpt_extra)
        torch.save(ckpt, self.softlink)
        wandb.save(self.softlink)

    def load_model(self, load_path: str):
        ckpt = torch.load(load_path, map_location="cpu")
        self.actor.load_state_dict(ckpt["spline_actor"])
        self.spline_optimizer = Adam(
            trainable_parameters(self.actor), lr=self.spline_lr, eps=1e-8
        )
        self.spline_optimizer.load_state_dict(ckpt["spline_optimizer"])
        self.ac.pi.load_state_dict(ckpt["controller_actor"])
        self.pi_optimizer = Adam(
            trainable_parameters(self.ac.pi), lr=self.pi_lr, eps=1e-8
        )
        self.pi_optimizer.load_state_dict(ckpt["pi_optimizer"])
        if ckpt["nagents"] == self.nagents:
            self.ac.v.load_state_dict(ckpt["controller_critic"])
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
                self.ac.v.load_state_dict(ckpt["controller_critic"])
                self.vf_optimizer.load_state_dict(ckpt["vf_optimizer"])
                self.logger.log(
                    "Agent doesn't depend on nagents. So continuing finetuning"
                )

    def dump_tabular(self):
        self.logger.log_tabular("Epoch", average_only=True)
        self.logger.log_tabular("EpisodeReturnSpline", with_min_and_max=True)
        self.logger.log_tabular("EntropySpline", average_only=True)
        self.logger.log_tabular("KLSpline", average_only=True)
        self.logger.log_tabular("StopIterSpline", average_only=True)
        self.logger.log_tabular("LossActorSpline", average_only=True)
        self.logger.log_tabular("EpisodeRunTimeSpline", average_only=True)
        self.logger.log_tabular("PPOUpdateTimeSpline", average_only=True)
        self.logger.log_tabular(
            "EpisodeReturnController", with_min_and_max=True
        )
        self.logger.log_tabular("EpisodeLengthController", average_only=True)
        self.logger.log_tabular("EntropyController", average_only=True)
        self.logger.log_tabular("KLController", average_only=True)
        self.logger.log_tabular("StopIterController", average_only=True)
        self.logger.log_tabular("ValueEstimateController", average_only=True)
        self.logger.log_tabular("LossActorController", average_only=True)
        self.logger.log_tabular("LossCriticController", average_only=True)
        self.logger.log_tabular("EpisodeRunTimeController", average_only=True)
        self.logger.log_tabular("PPOUpdateTimeController", average_only=True)
        self.logger.dump_tabular()

    def train(self):
        for epoch in range(self.epochs):
            self.logger.store(Epoch=epoch)
            start_time = time.time()
            self.controller_episode_runner()
            self.logger.store(
                EpisodeRunTimeController=time.time() - start_time
            )

            start_time = time.time()
            self.update_controller()
            self.logger.store(PPOUpdateTimeController=time.time() - start_time)

            start_time = time.time()
            self.spline_episode_runner()
            self.logger.store(EpisodeRunTimeSpline=time.time() - start_time)

            start_time = time.time()
            self.update_spline()
            self.logger.store(PPOUpdateTimeSpline=time.time() - start_time)

            if hvd.rank() == 0:
                self.save_model(epoch)

            self.dump_tabular()

    def controller_episode_runner(self):
        env = self.env
        (o, _), ep_ret, ep_len = env.reset(), 0, 0
        o, a_ids = env.step(
            0, self.actor.act(o.to(self.device), deterministic=True)
        )
        prev_done = torch.zeros(env.nagents, 1, device=self.device).bool()
        for t in range(self.local_steps_per_epoch):
            obs, lidar = o
            actions, val_f, log_probs = self.ac.step(
                [obs.to(self.device), lidar.to(self.device)]
            )
            (next_o, _a_ids), r, d, info = self.env.step(1, actions)

            ep_ret += r.mean()
            ep_len += 1

            done = d.all()

            for i, name in enumerate(a_ids):
                if prev_done[i]:
                    continue
                b = int(name.rsplit("_", 1)[-1])
                self.controller_buffer.store(
                    b,
                    obs[i],
                    lidar[i],
                    actions[i],
                    r[i],
                    val_f[i],
                    log_probs[i],
                )

            o = next_o
            a_ids = _a_ids
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
                self.controller_buffer.finish_path(v)

                if terminal:
                    self.logger.store(
                        EpisodeReturnController=ep_ret,
                        EpisodeLengthController=ep_len,
                    )
                (o, _), ep_ret, ep_len = env.reset(), 0, 0
                prev_done = torch.zeros(
                    env.nagents, 1, device=self.device
                ).bool()
                o, a_ids = env.step(
                    0, self.actor.act(o.to(self.device), deterministic=True)
                )

    def spline_episode_runner(self):
        env = self.env

        for _ in range(self.local_number_episodes):
            obs, _ = env.reset()
            _, actions, log_probs = self.actor(obs.to(self.device))
            o, _ = env.step(0, actions)

            done = False
            acc_reward = torch.zeros(
                self.nagents, 1, device=self.env.world.device
            )
            while not done:
                action = self.ac.act(
                    [t.to(self.device) for t in o], deterministic=True
                )
                (o, _), r, d, _ = env.step(1, action)
                done = d.all()
                acc_reward += r
            ep_ret = acc_reward.mean()

            self.spline_buffer.store(
                obs.detach(),
                actions.detach(),
                acc_reward[:, 0].to(self.device).detach(),
                log_probs.detach(),
            )
            self.logger.store(EpisodeReturnSpline=ep_ret)
