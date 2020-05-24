import numpy as np
import torch
import matplotlib
import torch.nn as nn

matplotlib.use("agg")
import matplotlib.pyplot as plt

EPS = 1e-7


class ActiveSplineTorch(nn.Module):
    def __init__(self, cp_num, p_num, alpha=0.5, device="cpu"):
        super(ActiveSplineTorch, self).__init__()
        self.cp_num = cp_num
        self.p_num = int(p_num / cp_num)
        self.alpha = alpha
        self.device = device

    def batch_arange(self, start_t, end_t, step_t):
        batch_arr = map(torch.arange, start_t, end_t, step_t)
        batch_arr = [arr.unsqueeze(0) for arr in batch_arr]
        return torch.cat(batch_arr, dim=0)

    def batch_linspace(self, start_t, end_t, step_t, device="cuda"):
        step_t = [step_t] * end_t.size(0)
        batch_arr = map(torch.linspace, start_t, end_t, step_t)
        batch_arr = [arr.unsqueeze(0) for arr in batch_arr]
        return torch.cat(batch_arr, dim=0).to(device)

    def forward(self, cps):
        return self.sample_point(cps)

    def sample_point(self, cps):
        cp_num = cps.size(1)
        cps = torch.cat([cps, cps[:, 0, :].unsqueeze(1)], dim=1)
        auxillary_cps = torch.zeros(
            cps.size(0),
            cps.size(1) + 2,
            cps.size(2),
            device=cps.device,
            dtype=torch.float,
        )
        auxillary_cps[:, 1:-1, :] = cps

        l_01 = torch.sqrt(
            torch.sum(torch.pow(cps[:, 0, :] - cps[:, 1, :], 2), dim=1) + EPS
        )
        l_last_01 = torch.sqrt(
            torch.sum(torch.pow(cps[:, -1, :] - cps[:, -2, :], 2), dim=1) + EPS
        )

        l_01.detach_().unsqueeze_(1)
        l_last_01.detach_().unsqueeze_(1)

        auxillary_cps[:, 0, :] = cps[:, 0, :] - l_01 / l_last_01 * (
            cps[:, -1, :] - cps[:, -2, :]
        )
        auxillary_cps[:, -1, :] = cps[:, -1, :] + l_last_01 / l_01 * (
            cps[:, 1, :] - cps[:, 0, :]
        )

        t = torch.zeros(
            [auxillary_cps.size(0), auxillary_cps.size(1)],
            device=cps.device,
            dtype=torch.float,
        )
        for i in range(1, t.size(1)):
            t[:, i] = (
                torch.pow(
                    torch.sqrt(
                        torch.sum(
                            torch.pow(
                                auxillary_cps[:, i, :]
                                - auxillary_cps[:, i - 1, :],
                                2,
                            ),
                            dim=1,
                        )
                    ),
                    self.alpha,
                )
                + t[:, i - 1]
            )

        # No need to calculate gradient w.r.t t.
        t = t.detach()
        # print(t)
        lp = 0
        points = torch.zeros(
            [cps.size(0), self.p_num * self.cp_num, cps.size(2)],
            device=cps.device,
            dtype=torch.float,
        )

        for sg in range(1, self.cp_num + 1):
            v = self.batch_linspace(
                t[:, sg], t[:, sg + 1], self.p_num, cps.device
            )
            t0 = t[:, sg - 1].unsqueeze(1)
            t1 = t[:, sg].unsqueeze(1)
            t2 = t[:, sg + 1].unsqueeze(1)
            t3 = t[:, sg + 2].unsqueeze(1)

            for i in range(self.p_num):
                tv = v[:, i].unsqueeze(1)
                x01 = (t1 - tv) / (t1 - t0) * auxillary_cps[:, sg - 1, :] + (
                    tv - t0
                ) / (t1 - t0) * auxillary_cps[:, sg, :]
                x12 = (t2 - tv) / (t2 - t1) * auxillary_cps[:, sg, :] + (
                    tv - t1
                ) / (t2 - t1) * auxillary_cps[:, sg + 1, :]
                x23 = (t3 - tv) / (t3 - t2) * auxillary_cps[:, sg + 1, :] + (
                    tv - t2
                ) / (t3 - t2) * auxillary_cps[:, sg + 2, :]
                x012 = (t2 - tv) / (t2 - t0) * x01 + (tv - t0) / (
                    t2 - t0
                ) * x12
                x123 = (t3 - tv) / (t3 - t1) * x12 + (tv - t1) / (
                    t3 - t1
                ) * x23
                points[:, lp] = (t2 - tv) / (t2 - t1) * x012 + (tv - t1) / (
                    t2 - t1
                ) * x123
                lp = lp + 1

        return points
