from typing import Any, Optional, Sequence, Union, Dict
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import jax.numpy as jnp

import imitation
import imitation.environments
from imitation.algorithms.offline_rl.utils.datasets import MyDataset

from .base import BaseDataset


@dataclass(kw_only=True)
class CustomDataset(BaseDataset):

    dataset_impl: Any
    env: Any

    def get_eval_env(self, episode_length: int, action_repeat: int):
        return self.env.wrap_for_eval(episode_length, action_repeat)

    def sample_episodes(self, batch_size):
        return self.dataset_impl.sample(batch_size)

    def sample(self, key, batch_size):
        return self.dataset_impl.sample(key, batch_size)

    @staticmethod
    def from_resource_path(path, env_name):
        full_path = files(imitation) / path
        data = jnp.load(full_path)
        dataset = MyDataset.create(**data)
        env = getattr(imitation.environments, env_name)
        return CustomDataset(dataset_impl=dataset, env=env)
