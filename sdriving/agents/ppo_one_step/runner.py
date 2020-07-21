from typing import Optional

import torch
import wandb
from spinup.utils.mpi_tools import proc_id


def episode_runner(
    total_timesteps: int, device, buffer, env, actor: torch.nn.Module, logger,
):
    o, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_timesteps):
        a, logp = {}, {}

        for a_id, obs in o.items():
            pi = actor._distribution(obs.to(device))
            a[a_id] = actor.sample(pi)
            logp[a_id] = actor._log_prob_from_distribution(pi, a[a_id])
            a[a_id] = a[a_id].cpu()

        _, r, _, _ = env.step(a)

        epret = sum(r.values()).detach().sum() / len(r.keys())

        for a_id, obs in o.items():
            buffer.store(
                obs.cpu().detach(),
                a[a_id].cpu().detach(),
                torch.as_tensor(r[a_id]).detach().cpu(),
                logp[a_id].cpu().detach(),
            )

        logger.store(EpRet=epret)
        if proc_id() == 0:
            wandb.log({"Episode Return (Train)": epret})

        o, ep_ret, ep_len = env.reset(), 0, 0
