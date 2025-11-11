from typing import Any, Optional, Dict
from functools import partial
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp
from flax import struct

import gymnax
from gymnax.environments.spaces import Discrete
from gymnax.environments.environment import EnvState
from gymnax.wrappers import FlattenObservationWrapper

from .base import Env, BaseState, EvalMetrics
from imitation.environments import jax_acting as acting


class BaseWrapper:
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env):
        self.env = env

    def reset(self, rng: jax.Array):
        return self.env.reset(rng)

    def step(self, rng: jax.Array, state, action: jax.Array):
        return self.env.step(rng, state, action)

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)


class EpisodeWrapper(BaseWrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
      super().__init__(env)
      self.episode_length = episode_length
      self.action_repeat = action_repeat

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        return state

    def step(self, rng: jax.Array, state, action: jax.Array):
        def f(state, _):
            nstate = self.env.step(rng, state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info['episode_done']
        state.info['episode_metrics']['sum_reward'] += jnp.sum(rewards, axis=0)
        state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
        state.info['episode_metrics']['length'] += self.action_repeat
        state.info['episode_metrics']['length'] *= (1 - prev_done)
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
                state.info['episode_metrics'][metric_name] *= (1 - prev_done)

        state.info['episode_done'] = done
        return state.replace(done=done)


class VmapWrapper(BaseWrapper):
    """Vectorizes an environment."""

    def __init__(self, env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array):
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, rng: jax.Array, state, action: jax.Array):
        rng = jax.random.split(rng, action.shape[0])
        return jax.vmap(self.env.step)(rng, state, action)


class EvalWrapper(BaseWrapper):
    """Env with eval metrics."""

    def reset(self, rng: jax.Array):
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(
                jnp.zeros_like, reset_state.metrics
            ),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state

    def step(self, rng: jax.Array, state, action: jax.Array):
        state_metrics = state.info['eval_metrics']
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(
                f'Incorrect type for state_metrics: {type(state_metrics)}'
            )
        del state.info['eval_metrics']
        nstate = self.env.step(rng, state, action)
        nstate.metrics['reward'] = nstate.reward
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info['steps'],
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)

        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info['eval_metrics'] = eval_metrics
        return nstate


class AutoResetWrapper(BaseWrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array):
      state = self.env.reset(rng)
      state.info['first_state_impl'] = state.state_impl
      state.info['first_obs'] = state.obs
      return state

    def step(self, rng: jax.Array, state, action: jax.Array):
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(rng, state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        state_impl = jax.tree.map(
            where_done, state.info['first_state_impl'], state.state_impl
        )
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        return state.replace(state_impl=state_impl, obs=obs)


@struct.dataclass
class GymnaxState(BaseState):
    """Environment state for training and inference."""
    state_impl: EnvState
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


@dataclass
class GymnaxEnv(Env):

    env_params: Optional[any]

    def reset(self, rng: jax.Array):
        obs, state = self.env_impl.reset(rng, self.env_params)
        rew = jnp.zeros((), dtype=jnp.float32)
        done = jnp.zeros((), dtype=jnp.float32)
        return GymnaxState(state_impl=state, obs=obs, reward=rew, done=done, metrics={}, info={})

    def step(self, rng: jax.Array, state: GymnaxState, action: jax.Array):
        n_obs, n_state, reward, done, info = self.env_impl.step(rng, state.state_impl, action)
        # in gymnax termination/truncation is still one thing ("done")
        return GymnaxState(
            state_impl=n_state,
            obs=n_obs,
            reward=reward.astype(jnp.float32),
            done=done.astype(jnp.float32),
            metrics=state.metrics,
            info=state.info)

    def wrap_for_training(self, episode_length: int, action_repeat: int):
        wrapped_env_impl = FlattenObservationWrapper(self.env_impl)
        env = GymnaxEnv(env_impl=wrapped_env_impl, env_params=self.env_params)
        env = VmapWrapper(env)
        env = EpisodeWrapper(env, episode_length, action_repeat)
        env = AutoResetWrapper(env)
        return env

    def wrap_for_eval(self, episode_length: int, action_repeat: int):
        wrapped_env_impl = FlattenObservationWrapper(self.env_impl)
        env = GymnaxEnv(env_impl=wrapped_env_impl, env_params=self.env_params)
        env = VmapWrapper(env)
        env = EpisodeWrapper(env, episode_length, action_repeat)
        env = AutoResetWrapper(env)
        env = EvalWrapper(env)
        return env

    def wrap_for_visualization(self):
        wrapped_env_impl = FlattenObservationWrapper(self.env_impl)
        return GymnaxEnv(env_impl=wrapped_env_impl, env_params=self.env_params)

    def rollout(self, policy, n_episodes: int, episode_length: int, seed: int):
        env = self.wrap_for_eval(episode_length, 1)

        key = jax.random.PRNGKey(seed)
        reset_keys = jax.random.split(key, n_episodes)
        first_state = env.reset(reset_keys)
        data = acting.generate_unroll(
            env,
            first_state,
            policy,
            key,
            unroll_length=episode_length,
        )[1]

        return {
            'observation': np.asarray(data.observation),
            'action': np.asarray(data.action),
            'reward': np.asarray(data.reward),
            'next_observation': np.asarray(data.next_observation),
            'discount': np.asarray(data.discount),
        }

    @property
    def n_observations(self):
        obs_space = self.env_impl.observation_space(self.env_params)

        if isinstance(obs_space, Discrete):
            return obs_space.n
        else:
            raise NotImplementedError()

    @property
    def n_actions(self):
        action_space = self.env_impl.action_space(self.env_params)

        if isinstance(action_space, Discrete):
            return action_space.n
        else:
            raise NotImplementedError()

    @property
    def obs_size(self):
        obs_space = self.env_impl.observation_space(self.env_params)
        assert len(obs_space.shape) == 1, "Observation space is not flat"
        return obs_space.shape[0]

    @property
    def action_size(self):
        action_space = self.env_impl.action_space(self.env_params)
        assert len(action_space.shape) == 1, "Action space is not flat"
        return action_space.shape[0]

    @staticmethod
    def from_name(name):
        env_impl, env_params = gymnax.make(name)
        return GymnaxEnv(env_impl=env_impl, env_params=env_params)
