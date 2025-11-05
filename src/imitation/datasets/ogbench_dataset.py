from typing import Any, Optional, Sequence, Union, Dict
from dataclasses import dataclass

import ogbench

from .base import BaseDataset


@dataclass(kw_only=True)
class OGBenchDataset(BaseDataset):

    env_impl: Any
    train_dataset_impl: Any
    val_dataset_impl: Any

    # @property
    # def n_observations(self):
    #     obs_space = self.dataset_impl.observation_space

    #     if hasattr(obs_space, "n"):
    #         return obs_space.n
    #     else:
    #         raise NotImplementedError()

    # @property
    # def n_actions(self):
    #     action_space = self.dataset_impl.action_space

    #     if hasattr(action_space, "n"):
    #         return action_space.n
    #     else:
    #         raise NotImplementedError()

    # @property
    # def obs_size(self):
    #     obs_space = self.dataset_impl.observation_space
    #     assert len(obs_space.shape) == 1, "Observation space is not flat"
    #     return obs_space.shape[0]

    # @property
    # def action_size(self):
    #     action_space = self.dataset_impl.action_space
    #     assert len(action_space.shape) == 1, "Action space is not flat"
    #     return action_space.shape[0]

    def get_eval_env(self):
        return self.env_impl

    @staticmethod
    def from_name(name):
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(name)
        return OGBenchDataset(env_impl=env,
                              train_dataset_impl=train_dataset,
                              val_dataset_impl=val_dataset)
