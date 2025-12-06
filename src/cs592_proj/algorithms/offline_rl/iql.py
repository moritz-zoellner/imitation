# source https://github.com/ikostrikov/implicit_q_learning
# https://arxiv.org/abs/2110.06169


import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Sequence, Union

# import os
# import time
# from functools import partial
# from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import gradients
from brax.training import replay_buffers
from brax.training import distribution
from brax.training import networks
from brax.training import pmap
from brax.training.networks import ActivationFn, Initializer, FeedForwardNetwork, MLP
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from cs592_proj.environments.jax_acting import Evaluator


# import d4rl
# import distrax
# import flax
# import flax.linen as nn
# import gym
# import jax
# import jax.numpy as jnp
# import numpy as np
# import optax
# import tqdm
# import wandb
# from flax.training.train_state import TrainState
# from omegaconf import OmegaConf
# from pydantic import BaseModel

# os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

Params = Any
Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


#
# Q NETWORK IMPLEMENTATION
#

@flax.struct.dataclass
class IQLNetworks:
    q_network: networks.FeedForwardNetwork
    v_network: networks.FeedForwardNetwork


def make_discrete_q_network(
        obs_size: int,
        n_actions: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu,
        final_activation: ActivationFn = lambda x: x,
        n_critics: int = 2
) -> FeedForwardNetwork:
    """Creates a q function for discrete action spaces."""

    # Q function for discrete space of actions.
    class DiscreteQModule(linen.Module):
        """Q Module."""
        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray):
            hidden = jnp.concatenate([obs], axis=-1)
            res = []
            for _ in range(self.n_critics):
                critic_action_qs = MLP(
                    layer_sizes=list(hidden_layer_sizes) + [n_actions],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform()
                )(hidden)
                critic_action_qs = final_activation(critic_action_qs)
                res.append(critic_action_qs)

            return jnp.stack(res, axis=-1)

    q_module = DiscreteQModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs), apply=apply)


def make_v_network(
        obs_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu,
        final_activation: ActivationFn = lambda x: x,
) -> FeedForwardNetwork:
    """Creates a q function for discrete action spaces."""

    class VModule(linen.Module):
        """Value Function Module."""

        @linen.compact
        def __call__(self, obs: jnp.ndarray):
            hidden = jnp.concatenate([obs], axis=-1)
            critic_action_v = MLP(
                layer_sizes=list(hidden_layer_sizes) + [1],
                activation=activation,
                kernel_init=jax.nn.initializers.lecun_uniform()
            )(hidden)
            critic_action_v = final_activation(critic_action_v)
            return critic_action_v

    v_module = VModule()

    def apply(processor_params, v_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return v_module.apply(v_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: v_module.init(key, dummy_obs), apply=apply)


def get_make_policy_fn(iql_network: IQLNetworks, n_actions: int):

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> types.Policy:

        def random_policy(observation: types.Observation, key: PRNGKey) -> Tuple[types.Action, types.Extra]:
            action = jax.random.randint(key, (observation.shape[0],), 0, n_actions)
            return action, {}

        def greedy_policy(observation: types.Observation, key: PRNGKey) -> Tuple[types.Action, types.Extra]:
            qs = iql_network.q_network.apply(*params, observation)
            min_q = jnp.min(qs, axis=-1)
            action = min_q.argmax(axis=-1)
            return action, {}

        epsilon = jnp.float32(0.1)

        def eps_greedy_policy(observation: types.Observation, key: PRNGKey) -> Tuple[types.Action, types.Extra]:
            key_sample, key_coin = jax.random.split(key)
            coin_flip = jax.random.bernoulli(key_coin, 1 - epsilon)
            return jax.lax.cond(coin_flip, greedy_policy, random_policy, observation, key_sample)

        if deterministic:
            return greedy_policy
        else:
            return eps_greedy_policy

    return make_policy


def make_iql_networks(
        observation_size: int,
        n_actions: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: networks.ActivationFn = linen.elu) -> IQLNetworks:

    q_network = make_discrete_q_network(
        observation_size,
        n_actions,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation)

    v_network = make_v_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation)

    return IQLNetworks(q_network=q_network,
                       v_network=v_network)


#
# LOSS
#


def expectile_l2(u, expectile=0.8):
    weight = jnp.where(u > 0, expectile, (1 - expectile))
    return weight * (u**2)


def make_v_loss(
        iql_network: IQLNetworks,
        reward_scaling: float,
        discounting: float,
        expectile: float,
):
    """Creates the IQL losses."""
  
    v_network = iql_network.v_network
    q_network = iql_network.q_network
  
    def v_loss(
            v_params: Params,
            normalizer_params: Any,
            target_q_params: Params,
            transitions: dict,
            key: PRNGKey
    ) -> jnp.ndarray:
    
        # Q(s_t, a_t) for all actions
        double_qs = q_network.apply(normalizer_params, target_q_params, transitions["observations"])
        double_q = jax.vmap(lambda x, i: x.at[i].get())(double_qs, transitions["action"])
        q = jnp.min(double_q, axis=-1)
    
        v = v_network.apply(normalizer_params, v_params, transitions["observations"]).squeeze(-1)

        loss = expectile_l2(q - v, expectile=expectile).mean()
        return loss
  
    return v_loss


def make_q_loss(
        iql_network: IQLNetworks,
        reward_scaling: float,
        discounting: float,
        expectile: float,
):
    """Creates the IQL losses."""
  
    v_network = iql_network.v_network
    q_network = iql_network.q_network
  
    def q_loss(
            q_params: Params,
            normalizer_params: Any,
            v_params: Params,
            transitions: dict,
            key: PRNGKey
    ) -> jnp.ndarray:
  
        next_v = v_network.apply(normalizer_params, v_params, transitions["next_observations"]).squeeze(-1)
        target_q = jax.lax.stop_gradient(transitions["reward"] * reward_scaling +
                                        transitions["discount"] * discounting * next_v)

        # Q(s_t, a_t) for all actions
        double_qs = q_network.apply(normalizer_params, q_params, transitions["observations"])
        double_q = jax.vmap(lambda x, i: x.at[i].get())(double_qs, transitions["action"])

        error = double_q - jnp.expand_dims(target_q, -1)
    
        loss = 0.5 * jnp.mean(jnp.square(error))
        return loss
  
    return q_loss

#
# TRAINING
#

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    v_optimizer_state: optax.OptState
    v_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_map(lambda x: x[0], v)


def _init_training_state(
        key: PRNGKey,
        obs_size: int,
        local_devices_to_use: int,
        iql_network: IQLNetworks,
        q_optimizer: optax.GradientTransformation,
        v_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q, key_v = jax.random.split(key, 3)
    q_params = iql_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    v_params = iql_network.v_network.init(key_v)
    v_optimizer_state = v_optimizer.init(v_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.float32))

    training_state = TrainingState(
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        v_optimizer_state=v_optimizer_state,
        v_params=v_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        normalizer_params=normalizer_params)
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


@dataclass
class IQL:


    discounting: float = 0.9
    expectile: float = 0.8
    reward_scaling: float = 1.
    tau: float = 0.005
    num_envs: int = 256
    num_evals: int = 16
    min_replay_size: int = 0
    max_replay_size: int = 10_000
    normalize_observations: bool = True
    hidden_layer_sizes: Sequence[int] = (256, 256)
    learning_rate: float = 1e-4
    batch_size: int = 256
    grad_updates_per_step: int = 1
    num_eval_envs: int = 16
    episode_length: int = 1000
    action_repeat: int = 1
    deterministic_eval: bool = True
    checkpoint_logdir: Optional[str] = None

    def train_fn(self, *, run_config, dataset, progress_fn, **_):
        """IQL training"""

        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        device_count = local_devices_to_use * jax.process_count()

        num_evals_after_init = max(self.num_evals - 1, 1)
        num_training_steps_per_epoch = -(
            # divide remaining steps across epochs before each evaluation
            -(run_config.num_timesteps // self.batch_size) // (num_evals_after_init)
        )

        #
        # ENV SET UP
        #

        env = dataset.get_eval_env(episode_length=self.episode_length, action_repeat=self.action_repeat)
        obs_size = env.obs_size

        #
        # Q NETWORK SET UP
        #

        normalize_fn = lambda x, y: x
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize

        network_factory = make_iql_networks
        iql_network = network_factory(
            observation_size=obs_size,
            n_actions=env.n_actions,
            preprocess_observations_fn=normalize_fn,
            hidden_layer_sizes=self.hidden_layer_sizes,
        )
        make_policy = get_make_policy_fn(iql_network, n_actions=env.n_actions)

        q_optimizer = optax.adam(learning_rate=self.learning_rate)
        v_optimizer = optax.adam(learning_rate=self.learning_rate)

        # value function
        v_loss = make_v_loss(
            iql_network,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting,
            expectile=self.expectile,
        )
        v_update = gradients.gradient_update_fn(
            v_loss, v_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

        # q function
        q_loss = make_q_loss(
            iql_network,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting,
            expectile=self.expectile,
        )
        q_update = gradients.gradient_update_fn(
            q_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

        def sgd_step(
            carry: Tuple[TrainingState, PRNGKey],
            data: dict[str, jax.Array]
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry

            key, key_q, key_v = jax.random.split(key, 3)

            v_loss, new_v_params, v_optimizer_state = v_update(
                training_state.v_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                data,
                key_v,
                optimizer_state=training_state.v_optimizer_state)

            q_loss, new_q_params, q_optimizer_state = q_update(
                training_state.q_params,
                training_state.normalizer_params,
                new_v_params,
                data,
                key_q,
                optimizer_state=training_state.q_optimizer_state)

            new_target_q_params = jax.tree_map(lambda x, y: x * (1 - self.tau) + y * self.tau,
                                               training_state.target_q_params, new_q_params)

            metrics = {
                'v_loss': v_loss,
                'q_loss': q_loss,
            }

            new_training_state = TrainingState(
                q_optimizer_state=q_optimizer_state,
                q_params=new_q_params,
                target_q_params=new_target_q_params,
                v_optimizer_state=v_optimizer_state,
                v_params=new_v_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                normalizer_params=training_state.normalizer_params)
            return (new_training_state, key), metrics

        def training_step(
                training_state: TrainingState,
                key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            experience_key, training_key = jax.random.split(key)

            # sample data
            data = dataset.sample(key, self.batch_size)

            # update normalizer params
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data["observations"],
                pmap_axis_name=_PMAP_AXIS_NAME)
            training_state = training_state.replace(
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + self.batch_size
            )

            # Change the front dimension of transitions so 'update_step' is called
            # grad_updates_per_step times by the scan.
            data = jax.tree_map(lambda x: jnp.reshape(x, (self.grad_updates_per_step, -1) + x.shape[1:]), data)
            (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), data)

            return training_state, metrics

        def training_epoch(
                training_state: TrainingState,
                key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

            def f(carry, unused_t):
                ts, k = carry
                k, new_key = jax.random.split(k)
                ts, metrics = training_step(ts, k)
                return (ts, new_key), metrics

            (training_state, key), metrics = jax.lax.scan(
                f,
                (training_state, key),
                (),
                length=num_training_steps_per_epoch
            )
            metrics = jax.tree_map(jnp.mean, metrics)
            return training_state, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # Note that this is NOT a pure jittable method.
        def training_epoch_with_timing(
                training_state: TrainingState,
                key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            nonlocal training_walltime
            t = time.time()
            (training_state, metrics) = training_epoch(
                training_state, key
            )
            metrics = jax.tree_map(jnp.mean, metrics)
            jax.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                **{f'training/{name}': value for name, value in metrics.items()}
            }
            return training_state, metrics

        global_key, local_key = jax.random.split(jax.random.PRNGKey(run_config.seed))
        local_key = jax.random.fold_in(local_key, process_id)

        # Training state init
        training_state = _init_training_state(
            key=global_key,
            obs_size=obs_size,
            local_devices_to_use=local_devices_to_use,
            iql_network=iql_network,
            q_optimizer=q_optimizer,
            v_optimizer=v_optimizer)
        del global_key

        local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

        evaluator = Evaluator(
            env, # already wrapped for eval
            functools.partial(make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key=eval_key
        )

        # Run initial eval
        if process_id == 0 and self.num_evals > 1:
            params = _unpmap((training_state.normalizer_params, training_state.q_params))
            metrics = evaluator.run_evaluation(params, training_metrics={})
            logging.info(metrics)
            progress_fn(0, metrics, params=params)

        training_walltime = 0.0
        current_step = 0
        for _ in range(num_evals_after_init):
            logging.info('step %s', current_step)

            # Optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, training_metrics) = training_epoch_with_timing(
                training_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            # Eval and logging
            if process_id == 0:
                # Save current policy.
                params = _unpmap((training_state.normalizer_params, training_state.q_params))

                # Run evals.
                metrics = evaluator.run_evaluation(params, training_metrics)
                logging.info(metrics)
                progress_fn(current_step, metrics, params=params)

        total_steps = current_step
        assert total_steps >= run_config.num_timesteps

        params = _unpmap((training_state.normalizer_params, training_state.q_params))

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        logging.info('total steps: %s', total_steps)
        pmap.synchronize_hosts()
        return (make_policy, params, metrics)
