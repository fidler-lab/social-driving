import gym
import torch
from sdriving.agents.model import (
    PPOLidarActorCritic,
    PPOWaypointActorCritic,
    PPOWaypointCategoricalActor,
    PPOWaypointGaussianActor,
)


def checkpoint_parser(path):
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["model"] == "centralized_critic":
        centralized = True
        actor = PPOLidarActorCritic(
            **ckpt["ac_kwargs"], centralized=centralized
        )
        actor.v = None
        actor.pi.load_state_dict(ckpt["actor"])
        return actor
