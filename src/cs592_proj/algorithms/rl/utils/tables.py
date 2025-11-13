"""Table definitions."""

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

import flax
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class Table:
    init: Callable[..., Any]
    apply: Callable[..., Any]


@flax.struct.dataclass
class QLTables:
    q_table: Table


def get_make_policy_fn(ql_tables: QLTables):

    def make_policy(
        params: jax.Array,
        deterministic: bool = False,
    ):

        q_params = params

        def random_action_policy(observation, option_key):
            option = jax.random.randint(option_key, (observation.shape[0],), 0, n_options)
            return option, {}

        def greedy_policy(observation, option_key):
            qs = ql_tables.q_table.apply(q_params, observation)
            option = qs.argmax(axis=-1)
            return option, {}

        epsilon = jnp.float32(0.1)

        def eps_greedy_policy(observation, key_sample):
            key_sample, key_coin = jax.random.split(key_sample)
            coin_flip = jax.random.bernoulli(key_coin, 1 - epsilon)
            return jax.lax.cond(coin_flip, greedy_policy, random_option_policy, observation, key_sample)

        if deterministic:
            return greedy_policy
        else:
            return eps_greedy_policy

    return make_policy


def make_q_table(
        observation_space_size: int,
        action_space_size: int
):
    """Creates a q function table."""

    # Q function
    class QTable():
        """Q Table."""

        def __call__(self, params, obs: jnp.ndarray):
            return jnp.squeeze(params[obs])

    q_table = QTable()

    def init(key):
        return jnp.zeros((observation_space_size, action_space_size))

    def apply(q_params, obs):
        return q_table(q_params, obs)

    return Table(init=init, apply=apply)


def make_q_learning_tables(
    n_observations: int,
    n_actions: int,
) -> QLTables:
    q_table = make_q_table(n_observations, n_actions)
    
    return QLTables(q_table=q_table)
