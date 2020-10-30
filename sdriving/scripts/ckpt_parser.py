import gym
import torch

from sdriving.agents.model import (
    PPOLidarActorCritic,
    PPOWaypointActorCritic,
    PPOWaypointCategoricalActor,
    PPOWaypointGaussianActor,
)


def checkpoint_parser(path: str) -> tuple:
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["model"] == "centralized_critic":
        # PPO_OneStep
        if "type" in ckpt and ckpt["type"] == "one_step_ppo":
            params = ckpt["actor_kwargs"]
            if isinstance(params["act_space"], gym.spaces.Discrete):
                model = PPOWaypointCategoricalActor
            else:
                model = PPOWaypointGaussianActor
            actor = model(**params)
            actor.load_state_dict(ckpt["actor"])
            return actor, "PPO_ONESTEP"

        # PPO_Alternating_Optimization_Centralized_Critic
        if "type" in ckpt and ckpt["type"] == "bilevel_model":
            params = ckpt["actor_kwargs"]
            if isinstance(params["act_space"], gym.spaces.Discrete):
                model = PPOWaypointCategoricalActor
            else:
                model = PPOWaypointGaussianActor
            actor = model(**params)
            actor.load_state_dict(ckpt["spline_actor"])
            ac = PPOLidarActorCritic(**ckpt["ac_kwargs"])
            ac.v = None
            ac.pi.load_state_dict(ckpt["controller_actor"])
            return (
                (ac, actor),
                "PPO_ALTERNATING_OPTIMIZATION_CENTRALIZED_CRITIC",
            )

        # PPO_Centralized_Critic
        centralized = True
        ac = PPOLidarActorCritic(**ckpt["ac_kwargs"], centralized=centralized)
        ac.v = None
        ac.pi.load_state_dict(ckpt["actor"])
        return ac.pi, "PPO_CENTRALIZED_CRITIC"
