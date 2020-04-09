from typing import List

import torch
from mpc import mpc


class MPCController:
    def __init__(
        self,
        nb_actions: int = 2,
        nb_states: int = 4,
        slew_rate_penalty: float = 10.0,
        goal_weights: List[float] = [1.0] * 4,
        ctrl_penalty: float = 0.001,
        action_low: List[float] = [-0.1, -1.5],
        action_high: List[float] = [0.1, 1.5],
        nbatch: int = 1,
    ):
        assert len(goal_weights) == nb_states

        self.goal_weights = torch.as_tensor(goal_weights)
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.nbatch = nbatch
        self.slew_rate_penalty = slew_rate_penalty

        self.action_low = torch.as_tensor(action_low)[None, None, :].repeat(
            1, nbatch, 1
        )
        self.action_high = torch.as_tensor(action_high)[None, None, :].repeat(
            1, nbatch, 1
        )

        self.q = torch.cat(
            (self.goal_weights, ctrl_penalty * torch.ones(nb_actions))
        )

        self.uinit = None
        self.u_upper = None
        self.u_lower = None
        self.device = torch.device("cpu")

    def to(self, device):
        if device == self.device:
            return
        self.goal_weights = self.goal_weights.to(device)
        self.action_low = self.action_low.to(device)
        self.action_high = self.action_high.to(device)
        self.q = self.q.to(device)
        self.device = device

    def _get_cost_function(
        self, goal_state: torch.Tensor, timesteps: int = 10
    ):
        px = -torch.sqrt(self.goal_weights) * goal_state
        p = torch.cat(
            (px, torch.zeros(self.nbatch, self.nb_actions).to(self.device)),
            dim=-1,
        )
        Q = torch.diag(self.q).repeat(timesteps, self.nbatch, 1, 1)
        p = p.repeat(timesteps, 1, 1)
        return mpc.QuadCost(Q, p)

    def __call__(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        dynamics,
        lqr_iter: int = 5,
        timesteps: int = 10,
    ):
        assert start_state.size(0) == goal_state.size(0) == self.nbatch
        start_state = start_state.to(self.device)
        goal_state = goal_state.to(self.device)
        dynamics.to(self.device)

        if self.u_lower is None:
            self.u_lower = self.action_low.repeat(timesteps, 1, 1)
            self.u_upper = self.action_high.repeat(timesteps, 1, 1)

        # Using analytic gradients allow us to bypass the memory leak
        # bug in the mpc.pytorch code to some extent
        controller = mpc.MPC(
            self.nb_states,
            self.nb_actions,
            timesteps,
            u_lower=self.u_lower,
            u_upper=self.u_upper,
            lqr_iter=lqr_iter,
            eps=1e-2,
            n_batch=self.nbatch,
            backprop=False,
            verbose=-1,
            grad_method=mpc.GradMethods.ANALYTIC,
            exit_unconverged=False,
            slew_rate_penalty=self.slew_rate_penalty,
        )

        cost_fn = self._get_cost_function(goal_state, timesteps=timesteps)

        # FIXME:
        # mpc-pytorch has a memory leak bug. Every call in this line will
        # lead to increase in total memory and eventually will lead to
        # OOM error
        # The bug is in the autodiff code. So if we provide the analytic
        # derivative, we should be fine to some extent. There is another
        # memory leak but I haven't been able to track it down
        nstates, nactions, nobjs = controller(start_state, cost_fn, dynamics)
        return nstates[:timesteps], nactions[:timesteps], nobjs[:timesteps]

    def reset(self):
        self.uinit = None
        self.u_upper = None
        self.u_lower = None
