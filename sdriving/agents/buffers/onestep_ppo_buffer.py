import torch

from sdriving.agents.utils import combined_shape


def allocate_zeros_tensor(size, device):
    return torch.zeros(size, dtype=torch.float32, device=device)


class OneStepPPOBuffer:
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        size: int,
        nagents: int = 1,
        device=torch.device("cpu"),
    ):
        func = lambda x: allocate_zeros_tensor(
            combined_shape(size, x, batch=nagents), device
        )

        self.state_buf = func(state_dim)
        self.act_buf = func(act_dim)

        func = lambda: allocate_zeros_tensor(
            combined_shape(size, batch=nagents), device
        )

        self.rew_buf = func()
        self.logp_buf = func()

        self.max_size = size
        self.ptr = 0

    def store(self, obs, act, rew, logp):
        """Append one timestep of agent-environment interaction to the
        buffer."""
        assert (
            self.ptr < self.max_size
        )  # buffer has to have room so you can store
        self.state_buf[:, self.ptr] = obs
        self.act_buf[:, self.ptr] = act
        self.rew_buf[:, self.ptr] = rew
        self.logp_buf[:, self.ptr] = logp
        self.ptr = self.ptr + 1

    def get(self):
        ptr_copy = self.ptr
        self.ptr = 0
        rew = self.rew_buf[:, :ptr_copy]
        rew = (rew - rew.mean(1, keepdim=True)) / (
            rew.std(1, keepdim=True) + 1e-7
        )

        return dict(
            obs=self.state_buf[:, :ptr_copy],
            act=self.act_buf[:, :ptr_copy],
            rew=rew,
            logp=self.logp_buf[:, :ptr_copy],
        )
