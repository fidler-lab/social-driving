from ray.rllib.env.multi_agent_env import MultiAgentEnv
from sdriving.environments import REGISTRY
import torch

class MultiAgentDrivingEnvironmentRLLibWrapper(MultiAgentEnv):
    def __init__(self, config):
        env_name = config["env_name"]
        args = config.get("args", [])
        kwargs = config.get("kwargs", {})

        self.env = REGISTRY[env_name](*args, **kwargs)

        self.action_space = self.env.get_action_space()
        self.observation_space = self.env.get_observation_space()

    @staticmethod
    def convert_sdriving_to_rllib_observation(obs, names):
        obs_dict = dict()
        if isinstance(obs, (list, tuple)):
            obs = [o.detach().cpu().numpy() for o in obs]
            for i, name in enumerate(names):
                obs_dict[name] = [o[i] for o in obs]
        else:
            obs = obs.detach().cpu().numpy()
            for i, name in enumerate(names):
                obs_dict[name] = obs[i]
        return obs_dict

    @staticmethod
    def convert_sdriving_to_rllib_reward(rewards, names):
        return {names[i]: rewards[i].item() for i in range(rewards.size(0))}

    def reset(self):
        observations, agent_names = self.env.reset()
        return self.convert_sdriving_to_rllib_observation(
            observations, agent_names
        )

    def step(self, actions, *args, **kwargs):
        act_list = []
        for name in self.env.agent_names_copy:
            if name in actions:
                act_list.append(torch.as_tensor(actions[name]).unsqueeze(0))
        actions = torch.cat(act_list, dim=0)

        observations, rewards, _dones, infos = self.env.step(actions, *args, **kwargs)
        # return observations, rewards, infos
        if _dones.all():
            return (
                {},
                self.convert_sdriving_to_rllib_reward(
                    rewards, self.env.agent_names_copy
                ),
                {"__all__": True},
                {}
            )

        dones = {"__all__": _dones.all().item()}
        dones.update({
            name: _dones[i].item()
            for i, name in enumerate(observations[1])
        })
        rewards = self.convert_sdriving_to_rllib_reward(
            rewards, observations[1]
        )
        observations = self.convert_sdriving_to_rllib_observation(
            observations[0], observations[1]
        )
        return observations, rewards, dones, infos