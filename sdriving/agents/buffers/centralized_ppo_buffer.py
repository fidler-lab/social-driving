import torch

from sdriving.agents.utils import (
    combined_shape,
    discount_cumsum,
    hvd_scalar_statistics,
)


def allocate_zeros_tensor(size, device):
    return torch.zeros(size, dtype=torch.float32, device=device)


def allocate_ones_tensor(size, device):
    return torch.ones(size, dtype=torch.float32, device=device)


class CentralizedPPOBuffer:
    def __init__(
        self,
        state_dim: int,
        lidar_dim: int,
        act_dim: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        nagents: int = 1,
        device=torch.device("cpu"),
    ):
        func = lambda x: allocate_zeros_tensor(
            combined_shape(size, x, batch=nagents), device
        )

        self.state_buf = func(state_dim)
        self.lidar_buf = func(lidar_dim)
        self.act_buf = func(act_dim)

        func = lambda: allocate_zeros_tensor(
            combined_shape(size, batch=nagents), device
        )

        self.vest_buf = func()
        self.adv_buf = func()
        self.rew_buf = func()
        self.ret_buf = func()
        self.val_buf = func()
        self.logp_buf = func()

        # We need to mask a few values
        self.mask_buf = allocate_ones_tensor(
            combined_shape(size, batch=nagents), device
        )

        self.gamma, self.lam = gamma, lam
        self.max_size = size
        self.nagents = nagents
        self.ptr = [0 for _ in range(nagents)]
        self.path_start_idx = [0 for _ in range(nagents)]

    def store(self, b: int, obs, lidar, act, rew, val, logp):
        """Append one timestep of agent-environment interaction to the
        buffer."""
        idx = self.ptr[b]
        assert idx < self.max_size

        self.state_buf[b, idx] = obs
        if lidar is not None:
            self.lidar_buf[b, idx] = lidar

        self.act_buf[b, idx] = act
        self.rew_buf[b, idx] = rew
        self.val_buf[b, idx] = val
        self.logp_buf[b, idx] = logp
        self.ptr[b] = idx + 1

    def finish_path(self, last_val: torch.Tensor):
        last_val = last_val.unsqueeze(1)
        max_ptr = max(self.ptr)
        for b in range(self.nagents):
            path_slice = slice(self.path_start_idx[b], self.ptr[b])
            rews = torch.cat([self.rew_buf[b, path_slice], last_val[b]])
            vals = torch.cat([self.val_buf[b, path_slice], last_val[b]])

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[b, path_slice] = discount_cumsum(
                deltas, self.gamma * self.lam
            )

            # the next line computes rewards-to-go,
            # to be targets for the value function
            self.ret_buf[b, path_slice] = discount_cumsum(rews, self.gamma)[
                :-1
            ]

            self.mask_buf[b, self.ptr[b] : max_ptr] = 0
            self.path_start_idx[b] = max_ptr
            self.ptr[b] = max_ptr

    def get(self):
        """Call this at the end of an epoch to get all of the data from the
        buffer, with advantages appropriately normalized (shifted to have mean
        zero and std one).
        Also, resets some pointers in the buffer.
        """
        mbuf = self.mask_buf.clone()
        torch.fill_(self.mask_buf, 1.0)
        for b in range(self.nagents):
            self.ptr[b], self.path_start_idx[b] = 0, 0

            # the next two lines implement the advantage normalization trick
            adv = self.adv_buf[b]
            # This is supposed to be computed across all processes but it
            # becomes a big bottleneck
            # adv_mean, adv_std = (
            #     adv.mean(),
            #     adv.std(),
            # )
            adv_mean, adv_std = hvd_scalar_statistics(self.adv_buf[b])
            self.adv_buf[b] = (adv - adv_mean) / (adv_std + 1e-7)
        # The entire buffer will most likely not be filled
        return dict(
            obs=self.state_buf,
            lidar=self.lidar_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
            vest=self.val_buf,
            mask=mbuf,
        )
