import gym
import torch
from sdriving.agents.model import (
    IterativeWayPointPredictor,
    PPOLidarActorCritic,
    PPOWaypointActorCritic,
    PPOWaypointCategoricalActor,
    PPOWaypointGaussianActor,
)


def checkpoint_parser(path):
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["type"] == "alt_opt":
        if isinstance(ckpt["actor_kwargs"]["act_space"], gym.spaces.Discrete):
            actor = PPOWaypointCategoricalActor(**ckpt["actor_kwargs"])
        else:
            actor = PPOWaypointGaussianActor(**ckpt["actor_kwargs"])
        actor.load_state_dict(ckpt["spline"])
        centralized = ckpt["model"] == "centralized_critic"
        ac = PPOLidarActorCritic(**ckpt["ac_kwargs"], centralized=centralized)
        ac.v = None
        ac.pi.load_state_dict(ckpt["actor"])
        return actor, ac
    elif ckpt["model"] == "reinforce" or (
        ckpt["model"] == "centralized_critic"
        and ckpt["type"] == "one_step_ppo"
    ):
        if isinstance(ckpt["actor_kwargs"]["act_space"], gym.spaces.Discrete):
            actor = PPOWaypointCategoricalActor(**ckpt["actor_kwargs"])
        else:
            actor = PPOWaypointGaussianActor(**ckpt["actor_kwargs"])
        actor.load_state_dict(ckpt["actor"])
        return actor
    else:
        centralized = ckpt["model"] == "centralized_critic"
        spline = "type" in ckpt and ckpt["type"] == "spline"
        if spline:
            actor = PPOWaypointActorCritic(
                **ckpt["ac_kwargs"], centralized=centralized
            )
        else:
            actor = PPOLidarActorCritic(
                **ckpt["ac_kwargs"], centralized=centralized
            )
        actor.v = None
        actor.pi.load_state_dict(ckpt["actor"])
        return actor
