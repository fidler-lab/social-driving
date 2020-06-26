import math

import scipy.special as sc
import torch
from sdriving.trafficsim.utils import angle_normalize
from torch import nn

EPS = 1e-7


class ClothoidMotion(nn.Module):
    # The approximations are from the paper
    # http://www.dgp.toronto.edu/~mccrae/projects/clothoid/sbim2008mccrae.pdf
    @staticmethod
    def _r(t):
        return (0.506 * t + 1) / (1.79 * (t ** 2) + 2.054 * t + 1.414)

    @staticmethod
    def _a(t):
        return 1 / (0.803 * (t ** 3) + 1.886 * (t ** 2) + 2.524 * t + 2)

    @staticmethod
    def _C(rt, phi):
        return 0.5 - rt * torch.sin(phi)

    @staticmethod
    def _S(rt, phi):
        return 0.5 - rt * torch.cos(phi)

    @staticmethod
    def _vector_from_angle(theta):
        if theta.ndim == 1:
            vec = torch.as_tensor([torch.cos(theta), torch.sin(theta)])
        else:
            vec = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        return vec / (vec.norm(0) + 1e-9)

    def forward(self, s_0, a, t, theta, dir):
        # Direction of normal
        n_theta = dir * math.pi / 2 + theta
        normal = self._vector_from_angle(n_theta)
        # Direction of tangent
        t_theta = theta
        tangent = self._vector_from_angle(t_theta)

        rt = self._r(t / a)
        phi = 0.5 * math.pi * (self._a(t / a) - (t / a) ** 2)
        Ct = self._C(rt, phi)
        St = self._S(rt, phi)
        # St, Ct = sc.fresnel((t / a).detach().cpu().numpy())
        # Ct = torch.as_tensor(Ct)
        # St = torch.as_tensor(St)

        s_t = s_0 + a * (Ct * tangent + St * normal)
        theta = theta + math.pi / 2 * (t ** 2)
        return s_t, angle_normalize(theta)


class LinearSplineMotion(nn.Module):
    def forward(
        self, pt1: torch.Tensor, pt2: torch.Tensor, t: torch.Tensor,
    ):
        # pt1 --> N x 2
        # pt2 --> N x 2
        # t   --> N x 1
        return pt1 * (1 - t) + pt2 * t


class CatmullRomSplineMotion(nn.Module):
    def __init__(
        self,
        cps: torch.Tensor = torch.rand(2, 2),
        p_num: int = 100,
        alpha: float = 0.5,
        device="cpu",
    ):
        super().__init__()
        self.device = device

        cp_num = cps.size(0)
        cps = torch.cat([cps, cps[0, :].unsqueeze(0)], dim=0)
        auxillary_cps = torch.zeros(
            cps.size(0) + 2, cps.size(1), device=cps.device, dtype=torch.float,
        )
        auxillary_cps[1:-1, :] = cps

        l_01 = torch.sqrt(
            torch.sum(torch.pow(cps[0, :] - cps[1, :], 2), dim=0) + EPS
        )
        l_last_01 = torch.sqrt(
            torch.sum(torch.pow(cps[-1, :] - cps[-2, :], 2), dim=0) + EPS
        )

        l_01.detach_().unsqueeze_(0)
        l_last_01.detach_().unsqueeze_(0)

        auxillary_cps[0, :] = cps[0, :] - l_01 / l_last_01 * (
            cps[-1, :] - cps[-2, :]
        )
        auxillary_cps[-1, :] = cps[-1, :] + l_last_01 / l_01 * (
            cps[1, :] - cps[0, :]
        )

        t = torch.zeros(
            [auxillary_cps.size(0)], device=cps.device, dtype=torch.float,
        )
        diff = (
            (auxillary_cps[1:] - auxillary_cps[:-1])
            .pow(2)
            .sum(-1)
            .pow(alpha / 2)
        )
        t[1:] = torch.cumsum(diff, dim=0)

        # No need to calculate gradient w.r.t t.
        t = t.detach()

        self.t = t
        self.cps = cps
        self.cp_num = cp_num
        self.auxillary_cps = auxillary_cps
        self.device = device

        self.ts = torch.cat(
            [
                torch.linspace(self.t[i], self.t[i + 1] - EPS, steps=p_num)
                for i in range(1, len(self.t) - 2)
            ]
        )
        pts = self.forward(self.ts)
        diff = pts[1:] - pts[:-1]
        dist = torch.sqrt(diff.pow(2).sum(-1) + EPS)

        self.pts = pts
        self.diff = diff
        self.dist = dist
        self.curve_length = dist.sum().detach()
        self.arc_lengths = torch.cat(
            [torch.zeros(1), torch.cumsum(dist, dim=0)]
        )
        self.npoints = self.arc_lengths.size(0)

    def map_s_to_t(self, s, sgs=None):
        s = torch.reshape(s, (s.size(0),))
        tval = torch.zeros([s.size(0)], device=self.device, dtype=torch.float)

        s, idx = torch.sort(s)
        sg = 0
        for elem, (i, sv) in enumerate(zip(idx, s)):
            if sgs is None:
                search_forward = True
                while sv < 0:
                    sv = sv + self.arc_lengths[-1]
                    if elem == 0:
                        search_forward = False
                        sg = self.npoints - 1
                while (
                    sv > self.arc_lengths[(sg + 1) % self.npoints]
                    or sv < self.arc_lengths[sg]
                ):
                    if search_forward:
                        sg = (sg + 1) % self.npoints
                    if not search_forward:
                        sg = (sg - 1) % self.npoints
            else:
                sg = sgs[i]
            s0 = self.arc_lengths[sg]
            s1 = self.arc_lengths[(sg + 1) % self.npoints]
            t0 = self.ts[sg]
            t1 = self.ts[(sg + 1) % self.npoints]

            tval[i] = t0 + (sv - s0) * (t1 - t0) / (s1 - s0)

        return tval

    def forward(self, t):
        t = torch.reshape(t, (t.size(0),))
        points = torch.zeros(
            [t.size(0), self.cps.size(1)],
            device=self.device,
            dtype=torch.float,
        )

        t, idx = torch.sort(t)
        sg = 1
        for i, tv in zip(idx, t):
            while (
                tv > self.t[sg + 1] or tv < self.t[sg]
            ) and sg < self.cp_num:
                sg += 1
            t0 = self.t[sg - 1].unsqueeze(0)
            t1 = self.t[sg].unsqueeze(0)
            t2 = self.t[sg + 1].unsqueeze(0)
            t3 = self.t[sg + 2].unsqueeze(0)

            x01 = (
                (t1 - tv) * self.auxillary_cps[sg - 1, :]
                + (tv - t0) * self.auxillary_cps[sg, :]
            ) / (t1 - t0)
            x12 = (
                (t2 - tv) * self.auxillary_cps[sg, :]
                + (tv - t1) * self.auxillary_cps[sg + 1, :]
            ) / (t2 - t1)
            x23 = (
                (t3 - tv) * self.auxillary_cps[sg + 1, :]
                + (tv - t2) * self.auxillary_cps[sg + 2, :]
            ) / (t3 - t2)
            x012 = ((t2 - tv) * x01 + (tv - t0) * x12) / (t2 - t0)
            x123 = ((t3 - tv) * x12 + (tv - t1) * x23) / (t3 - t1)
            points[i] = ((t2 - tv) * x012 + (tv - t1) * x123) / (t2 - t1)

        return points
