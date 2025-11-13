from typing import Any, Optional, Sequence, Union, Dict
from dataclasses import dataclass

import minari
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, SymbolicObsWrapper

import gymnasium
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

from .base import BaseDataset


# Must register ale_py environments
gymnasium.register_envs(ale_py)


class BaseWrapper:
    """Wraps an environment to allow modular transformations."""

    def __init__(self, dataset):
        self.dataset = dataset

    def sample_episodes(self, n: int):
        return self.dataset.sample_episodes(n)

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.dataset, name)


class MinigridWrapper(BaseWrapper):
    """Gives a reasonable default observation format for minigrid datasets."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.recovered_env = dataset.dataset_impl.recover_environment()
        self.wrapped_env = self.recovered_env # TODO

    @property
    def n_observations(self):
        breakpoint()
        obs_space = self.env_impl.observation_space(self.env_params)

        if isinstance(obs_space, Discrete):
            return obs_space.n
        else:
            raise NotImplementedError()

    @property
    def n_actions(self):
        breakpoint()
        action_space = self.env_impl.action_space(self.env_params)

        if isinstance(action_space, Discrete):
            return action_space.n
        else:
            raise NotImplementedError()

    @property
    def obs_size(self):
        breakpoint()
        obs_space = self.env_impl.observation_space(self.env_params)
        assert len(obs_space.shape) == 1, "Observation space is not flat"
        return obs_space.shape[0]

    @property
    def action_size(self):
        breakpoint()
        action_space = self.env_impl.action_space(self.env_params)
        assert len(action_space.shape) == 1, "Action space is not flat"
        return action_space.shape[0]

    def sample_episodes(self, n: int):
        breakpoint()
        pass


class AtariWrapper(BaseWrapper):
    """Gives a reasonable default observation format for atari datasets."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.recovered_env = dataset.dataset_impl.recover_environment()
        self.wrapped_env = AtariPreprocessing(self.recovered_env, frame_skip=1)
        # TODO: maybe FrameStackObservation

    @property
    def n_observations(self):
        breakpoint()
        obs_space = self.env_impl.observation_space(self.env_params)

        if isinstance(obs_space, Discrete):
            return obs_space.n
        else:
            raise NotImplementedError()

    @property
    def n_actions(self):
        breakpoint()
        action_space = self.env_impl.action_space(self.env_params)

        if isinstance(action_space, Discrete):
            return action_space.n
        else:
            raise NotImplementedError()

    @property
    def obs_size(self):
        breakpoint()
        obs_space = self.env_impl.observation_space(self.env_params)
        assert len(obs_space.shape) == 1, "Observation space is not flat"
        return obs_space.shape[0]

    @property
    def action_size(self):
        breakpoint()
        action_space = self.env_impl.action_space(self.env_params)
        assert len(action_space.shape) == 1, "Action space is not flat"
        return action_space.shape[0]

    def sample_episodes(self, n: int):
        breakpoint()
        pass


@dataclass(kw_only=True)
class MinariDataset(BaseDataset):

    dataset_impl: Any

    @property
    def n_observations(self):
        obs_space = self.dataset_impl.observation_space

        if hasattr(obs_space, "n"):
            return obs_space.n
        else:
            raise NotImplementedError()

    @property
    def n_actions(self):
        action_space = self.dataset_impl.action_space

        if hasattr(action_space, "n"):
            return action_space.n
        else:
            raise NotImplementedError()

    @property
    def obs_size(self):
        obs_space = self.dataset_impl.observation_space
        assert len(obs_space.shape) == 1, "Observation space is not flat"
        return obs_space.shape[0]

    @property
    def action_size(self):
        action_space = self.dataset_impl.action_space
        assert len(action_space.shape) == 1, "Action space is not flat"
        return action_space.shape[0]

    def get_eval_env(self):
        return self.dataset_impl.recover_environment()

    @staticmethod
    def from_name(name, wrappers=[]):
        dataset_impl = minari.load_dataset(name)

        dataset = MinariDataset(dataset_impl=dataset_impl)
        for wrapper in wrappers:
            dataset = wrapper(dataset)

        return dataset
