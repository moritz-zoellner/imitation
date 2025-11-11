from typing import Optional
from dataclasses import dataclass

import jax
import navix
from navix.spaces import Discrete

from .base import Env


def make_env_tabular(env):
    unique_id_obs_fn = env.get_unique_id_obs_fn()
    n_unique_states = env.get_n_unique_states()
    unique_id_obs_space = Discrete.create(n_elements=n_unique_states, shape=(1,))

    return env.replace(
        observation_fn=unique_id_obs_fn,
        observation_space=unique_id_obs_space,
    )

class BaseWrapper:
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env):
    self.env = env

  def reset(self, rng: jax.Array):
    return self.env.reset(rng)

  def step(self, state, action: jax.Array):
    return self.env.step(state, action)

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)


class VmapWrapper(BaseWrapper):
    """Vectorizes an environment."""

    def __init__(self, env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array):
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state, action: jax.Array):
        return jax.vmap(self.env.step)(state, action)


@dataclass
class NavixEnv(Env):

    def wrap_for_training(self, make_tabular: bool = False):
        if make_tabular:
            env = make_env_tabular(self.env)
        else:
            env = self.env
        wrapped_env = VmapWrapper(env)
        return NavixEnv(env=wrapped_env, env_params=self.env_params)

    @property
    def n_observations(self):
        obs_space = self.env.observation_space

        if isinstance(obs_space, Discrete):
            return obs_space.n
        else:
            raise NotImplementedError()

    @property
    def n_actions(self):
        action_space = self.env.action_space

        if isinstance(action_space, Discrete):
            return action_space.n
        else:
            raise NotImplementedError()

    @staticmethod
    def from_name(name):
        env = navix.make(name)
        return NavixEnv(env=env, env_params=None)
