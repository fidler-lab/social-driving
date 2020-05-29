import math

import scipy.special as sc
import torch
from torch import nn

from sdriving.trafficsim.utils import angle_normalize


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
