from typing import Any, Optional, Sequence, Union, Dict
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

import cs592_proj
import cs592_proj.environments

from .base import BaseDataset


class CustomDatasetImpl(FrozenDict):

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        if "observations" not in data and "observation" in data:
            data["observations"] = data["observation"]
            del data["observation"]
        if "next_observations" not in data and "next_observation" in data:
            data["next_observations"] = data["next_observation"]
            del data["next_observation"]
        assert 'observations' in data
        data = jax.tree_util.tree_map(lambda arr: jnp.array(arr), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = jnp.array(self._dict["observations"].shape[:-1])

    def get_random_idxs(self, key, num_idxs):
        """Return `num_idxs` random indices."""
        def f(key, upper_bound):
            return jax.random.randint(key, (num_idxs,), 0, upper_bound)

        keys = jax.random.split(key, self.size.shape[0])
        idxs = jax.vmap(f, in_axes=(0, 0))(keys, self.size)
        return idxs

    def sample(self, key, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(key, batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[(*idxs, ...)], self._dict)
        return result


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
        full_path = files(cs592_proj) / path
        data = jnp.load(full_path)
        dataset = CustomDatasetImpl.create(**data)
        env = getattr(cs592_proj.environments, env_name)
        return CustomDataset(dataset_impl=dataset, env=env)
