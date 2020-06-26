from typing import Optional

import torch
import wandb

from spinup.utils.mpi_tools import proc_id


def episode_runner(
    total_timesteps: int,
    device,
    buffer,
    env,
    ac: torch.nn.Module,
    logger,
    spline_model: Optional[torch.nn.Module] = None,
):
    o, ep_ret, ep_len = env.reset(), 0, 0

    if spline_model is not None:
        env.register_track(spline_model)

    for t in range(total_timesteps):
        a, v, logp, o_list = {}, {}, {}, []
        for k, obs in o.items():
            obs = tuple([t.detach().to(device) for t in obs])
            o[k] = obs
            o_list.append(obs)

        actions, val_f, log_probs = ac.step(o_list)
        for i, key in enumerate(o.keys()):
            a[key] = actions[i].cpu()
            v[key] = val_f
            logp[key] = log_probs[i]

        next_o, r, d, info = env.step(a)

        rlist = [torch.as_tensor(rwd).detach().cpu() for _, rwd in r.items()]
        ret = sum(rlist)
        rlen = len(rlist)
        ep_ret += ret / rlen
        ep_len += 1

        done = d["__all__"]

        for key, obs in o.items():
            buffer.store(
                key,
                obs[0].cpu(),
                obs[1].cpu(),
                a[key].cpu(),
                torch.as_tensor(r[key]).detach().cpu(),
                v[key].cpu(),
                logp[key].cpu(),
            )

        logger.store(VVals=val_f)

        o = next_o

        timeout = info["timeout"] if "timeout" in info else done
        terminal = done or timeout
        epoch_ended = t == total_timesteps - 1

        if terminal or epoch_ended:
            if epoch_ended and not terminal:
                o_list = []
                for _, obs in o.items():
                    o_list.append(tuple([t.to(device) for t in obs]))
                _, v, _ = ac.step(o_list)
                v = v.cpu()
            else:
                v = 0
            buffer.finish_path(v)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            if terminal:
                if proc_id() == 0:
                    wandb.log(
                        {
                            "Episode Return (Train)": ep_ret,
                            "Episode Length (Train)": ep_len,
                        }
                    )
            o, ep_ret, ep_len = env.reset(), 0, 0
            if spline_model is not None:
                env.register_track(spline_model)
