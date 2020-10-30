from typing import List

import torch
from torch import nn

from sdriving.tsim.utils import angle_normalize, remove_batch_element

EPS = 1e-7


@torch.jit.script
def batched_linspace(start: torch.Tensor, end: torch.Tensor, steps: int):
    steps = [steps] * end.size(0)
    batch_arr = [
        torch.linspace(start[i], end[i], steps[i])
        for i in range(start.size(0))
    ]
    return torch.cat(batch_arr, dim=0)


@torch.jit.script
def batched_2d_linspace(start: torch.Tensor, end: torch.Tensor, steps: int):
    return torch.cat(
        [
            batched_linspace(start[i, :], end[i, :], steps).unsqueeze(0)
            for i in range(start.size(0))
        ]
    )


class _CatmullRomSpline(nn.Module):
    def __init__(
        self,
        cps: torch.Tensor,  # N x P x 2
        p_num: int = 100,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.device = cps.device

        cp_num = cps.size(1)
        cps = torch.cat([cps, cps[:, 0:1, :]], dim=1)
        auxillary_cps = torch.cat(
            [
                torch.zeros(
                    cps.size(0),
                    1,
                    cps.size(2),
                    device=self.device,
                    dtype=torch.float,
                ),
                cps,
                torch.zeros(
                    cps.size(0),
                    1,
                    cps.size(2),
                    device=self.device,
                    dtype=torch.float,
                ),
            ],
            dim=1,
        )

        cps_01 = cps[:, 0, :] - cps[:, 1, :]
        cps_last_01 = cps[:, -1, :] - cps[:, -2, :]
        l_01 = (cps_01.pow(2).sum(1, keepdim=True) + EPS).sqrt().detach()
        l_last_01 = (
            (cps_last_01.pow(2).sum(1, keepdim=True) + EPS).sqrt().detach()
        )

        auxillary_cps[:, 0, :] = cps[:, 0, :] - l_01 / l_last_01 * cps_last_01
        auxillary_cps[:, -1, :] = cps[:, -1, :] - l_last_01 / l_01 * cps_01

        diff = (
            (auxillary_cps[:, 1:, :] - auxillary_cps[:, :-1, :])
            .pow(2)
            .sum(-1)
            .pow(alpha / 2)
        )

        t = torch.cat(
            [
                torch.zeros(
                    auxillary_cps.size(0),
                    1,
                    device=self.device,
                    dtype=torch.float,
                ),
                torch.cumsum(diff, dim=-1).detach(),
            ],
            dim=-1,
        )

        self.t = t  # N x ...
        self.cps = cps  # N x ...
        self.cp_num = cp_num
        self.p_num = p_num
        self.auxillary_cps = auxillary_cps  # N x ...

        self.ts = batched_2d_linspace(
            self.t[:, 1:-2], self.t[:, 2:-1] - 0.01, p_num
        ).reshape(
            cps.size(0), -1
        )  # N x T1
        pts = self.sample_points(self.ts)
        diff = pts[:, 1:, :] - pts[:, :-1, :]
        dist = (diff.pow(2).sum(-1) + EPS).sqrt()

        self.pts = pts  # N x T2 x 2
        self.diff = diff  # N x (T2 - 1) x 2
        self.dist = dist  # N x (T2 - 1)
        self.curve_length = dist.sum(-1).detach()  # N
        self.arc_lengths = torch.cat(
            [
                torch.zeros(cps.size(0), 1, device=self.device),
                torch.cumsum(dist, dim=-1),
            ],
            dim=1,
        )  # N x T2
        self.npoints = self.arc_lengths.size(1)

    @torch.jit.export
    def remove(self, idx: int):
        self.t = remove_batch_element(self.t, idx)
        self.cps = remove_batch_element(self.cps, idx)
        self.auxillary_cps = remove_batch_element(self.auxillary_cps, idx)
        self.ts = remove_batch_element(self.ts, idx)
        self.pts = remove_batch_element(self.pts, idx)
        self.diff = remove_batch_element(self.diff, idx)
        self.dist = remove_batch_element(self.dist, idx)
        self.curve_length = remove_batch_element(self.curve_length, idx)
        self.arc_lengths = remove_batch_element(self.arc_lengths, idx)

    @torch.jit.export
    def sample_points(self, t: torch.Tensor):  # t --> B x N
        # Assume that t is sorted along dim 1
        t = t.unsqueeze(2)  # B x N x 1
        ts = self.t.unsqueeze(1)

        c1, c2 = ts[:, :, 1:-2], ts[:, :, 2:-1]
        idx = torch.where((c1 <= t) * (t < c2))
        idx0 = idx[0]

        idx10 = idx[2]
        t0 = self.t[idx0, idx10].reshape(t.size())
        aux0 = self.auxillary_cps[idx0, idx10, :].reshape(
            t.size(0), t.size(1), 2
        )
        idx11 = idx10 + 1
        t1 = self.t[idx0, idx11].reshape(t.size())
        aux1 = self.auxillary_cps[idx0, idx11, :].reshape(
            t.size(0), t.size(1), 2
        )
        idx12 = idx11 + 1
        t2 = self.t[idx0, idx12].reshape(t.size())
        aux2 = self.auxillary_cps[idx0, idx12, :].reshape(
            t.size(0), t.size(1), 2
        )
        idx13 = idx12 + 1
        t3 = self.t[idx0, idx13].reshape(t.size())
        aux3 = self.auxillary_cps[idx0, idx12, :].reshape(
            t.size(0), t.size(1), 2
        )

        t0t, t1t, t2t, t3t = t0 - t, t1 - t, t2 - t, t3 - t

        x01 = (t1t * aux0 - t0t * aux1) / (t1 - t0)
        x12 = (t2t * aux1 - t1t * aux2) / (t2 - t1)
        x23 = (t3t * aux2 - t2t * aux3) / (t3 - t2)

        x012 = (t2t * x01 - t0t * x12) / (t2 - t0)
        x123 = (t3t * x12 - t1t * x23) / (t3 - t1)

        return (t2t * x012 - t1t * x123) / (t2 - t1)  # B x N x 2

    def forward(self, s: torch.Tensor, sgs: List[torch.Tensor]):
        # s --> B x N, sgs --> (B x N) X 2
        # s is sorted along dim 1
        i0, i1 = sgs[0], sgs[1]
        i2 = (sgs[1] + 1) % self.npoints
        s0 = self.arc_lengths[i0, i1].reshape(s.size())
        s1 = self.arc_lengths[i0, i2].reshape(s.size())
        t0 = self.ts[i0, i1].reshape(s.size())
        t1 = self.ts[i0, i2].reshape(s.size())

        return t0 + (s - s0) * (t1 - t0) / (s1 - s0)


def CatmullRomSpline(*args, **kwargs):
    return torch.jit.script(_CatmullRomSpline(*args, **kwargs))
