from typing import Optional

import torch
import wandb
from spinup.utils.mpi_tools import proc_id


def episode_runner(
    total_timesteps: int,
    device,
    buffer,
    env,
    actor: torch.nn.Module,
    ac: torch.nn.Module,
    logger,
    spline: bool,
):
    if spline:
        episode_runner_spline(
            total_timesteps, device, buffer, env, actor, ac, logger
        )
    else:
        episode_runner_controller(
            total_timesteps, device, buffer, env, actor, ac, logger
        )


def episode_runner_spline(
    total_timesteps: int,
    device,
    buffer,
    env,
    actor: torch.nn.Module,
    ac: torch.nn.Module,
    logger,
):
    for t in range(total_timesteps):
        env.reset()
        o = env.get_spline_state()
        a, logp = {}, {}

        for a_id, obs in o.items():
            pi = actor._distribution(obs.to(device))
            a[a_id] = actor.sample(pi)
            logp[a_id] = actor._log_prob_from_distribution(pi, a[a_id])
            a[a_id] = a[a_id].cpu()

        env.spline_step(a)
        rewards = {a_id: 0.0 for a_id in env.get_agent_ids_list()}

        done = False
        o2 = env.get_controller_state()
        while not done:
            a2 = {}
            for k, obs2 in o2.items():
                obs2 = tuple([t.detach().to(device) for t in obs2])
                a2[k] = ac.act(obs2, True).cpu()
            o2, r, dones, _ = env.controller_step(a2)
            rewards = {a_id: rwd + r[a_id] for (a_id, rwd) in rewards.items()}
            done = dones["__all__"]

        epret = torch.as_tensor(sum(rewards.values())).detach().sum() / len(
            r.keys()
        )

        for a_id, obs in o.items():
            buffer.store(
                obs.cpu().detach(),
                a[a_id].cpu().detach(),
                torch.as_tensor(rewards[a_id]).detach().cpu(),
                logp[a_id].cpu().detach(),
            )

        logger.store(EpRetSpline=epret)
        if proc_id() == 0:
            wandb.log({"Episode Return (Spline)": epret})


def episode_runner_controller(
    total_timesteps: int,
    device,
    buffer,
    env,
    actor: torch.nn.Module,
    ac: torch.nn.Module,
    logger,
):
    env.reset()
    ep_ret, ep_len = 0, 0
    o = env.get_controller_state()

    # Spline Action
    a2 = {}
    for a_id, obs2 in env.get_spline_state().items():
        a2[a_id] = actor.act(obs2.to(device), True).cpu()
    env.spline_step(a2)

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

        next_o, r, d, info = env.controller_step(a)

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

        logger.store(VValsController=val_f)

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

            logger.store(EpRetController=ep_ret, EpLenController=ep_len)
            if terminal:
                if proc_id() == 0:
                    wandb.log(
                        {
                            "Episode Return (Controller)": ep_ret,
                            "Episode Length (Controller)": ep_len,
                        }
                    )
            o, ep_ret, ep_len = env.reset(), 0, 0
            # Spline Action
            a2 = {}
            for a_id, obs2 in env.get_spline_state().items():
                a2[a_id] = actor.act(obs2.to(device), True).cpu()
            env.spline_step(a2)
