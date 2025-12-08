# https://arxiv.org/pdf/1606.03476

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

from brax.training.agents.ppo import losses as ppo_losses

from cs592_proj.environments import jax_acting as acting
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
class GAILNetworks:
    discriminator_network: networks.FeedForwardNetwork
    policy_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    value_network: networks.FeedForwardNetwork


class Categorical:
  """Categorical distribution over discrete actions."""

  def __init__(self, logits):
    """Initialize from unnormalized log-probabilities (logits).

    Args:
      logits: array of shape [..., num_actions]
    """
    self.logits = logits

  def sample(self, seed):
    """Sample integer actions with the same batch shape as logits[..., 0]."""
    return jax.random.categorical(seed, self.logits, axis=-1)

  def mode(self):
    """Most probable action (argmax over logits)."""
    return jnp.argmax(self.logits, axis=-1)

  def log_prob(self, x):
    """Log probability of integer actions x.

    Args:
      x: integer action indices with shape broadcastable to logits.shape[:-1].

    Returns:
      log_probs with shape equal to the broadcasted batch shape.
    """
    x = jnp.asarray(x, dtype=jnp.int32)
    log_probs = jax.nn.log_softmax(self.logits, axis=-1)
    # Expand x to select along the last dimension
    x_expanded = jnp.expand_dims(x, axis=-1)
    return jnp.take_along_axis(log_probs, x_expanded, axis=-1)[..., 0]

  def entropy(self):
    """Shannon entropy H[p] = -sum_a p(a) log p(a)."""
    log_probs = jax.nn.log_softmax(self.logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


class IdentityBijector:
  """Identity bijector."""

  def forward(self, x):
    return x

  def inverse(self, y):
    return y

  def forward_log_det_jacobian(self, x):
    # For discrete actions this is not really used, but we keep the interface.
    return jnp.zeros_like(jnp.asarray(x, dtype=jnp.float32))


class CategoricalDistribution(distribution.ParametricDistribution):
  """Parametric categorical distribution over discrete actions."""

  def __init__(self, num_actions):
    """Initialize the parametric distribution.

    Args:
      num_actions: number of discrete actions.
    """
    super().__init__(
        param_size=num_actions,            # one logit per action
        postprocessor=IdentityBijector(),  # no transform on actions
        event_ndims=0,                     # scalar discrete event
        reparametrizable=False,            # categorical is not reparametrizable
    )

  def create_dist(self, parameters):
    """Create a Categorical distribution from raw parameters.

    Args:
      parameters: array of shape [..., num_actions] interpreted as logits.

    Returns:
      A Categorical distribution instance.
    """
    logits = parameters
    return Categorical(logits=logits)


def make_discriminator_network(
        obs_size: int,
        n_actions: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a q function for discrete action spaces."""

    # Q function for discrete space of actions.
    class DiscreteActionDiscriminatorModule(linen.Module):
        """Discrete Action Discriminator Module."""

        @linen.compact
        def __call__(self, obs: jnp.ndarray):
            hidden = jnp.concatenate([obs], axis=-1)
            action_latents = MLP(
                layer_sizes=list(hidden_layer_sizes) + [n_actions],
                activation=activation,
                kernel_init=jax.nn.initializers.lecun_uniform()
            )(hidden)
            action_discriminations = linen.sigmoid(action_latents)
            return action_discriminations

    discriminator_module = DiscreteActionDiscriminatorModule()

    def apply(processor_params, discriminator_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return discriminator_module.apply(discriminator_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: discriminator_module.init(key, dummy_obs), apply=apply)


def get_make_policy_fn(gail_networks: GAILNetworks):

    parametric_action_distribution = gail_networks.parametric_action_distribution

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> types.Policy:

        normalizer_params, policy_params = params

        def policy(observations: types.Observation,
                   key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = gail_networks.policy_network.apply(normalizer_params, policy_params, observations)
            if deterministic:
                return parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions,
            }

        return policy

    return make_policy


def make_gail_networks(
        observation_size: int,
        n_actions: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        discriminator_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        activation: networks.ActivationFn = linen.elu,
        layer_norm: bool = False,
) -> GAILNetworks:

    parametric_action_distribution = CategoricalDistribution(n_actions)
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    discriminator_network = make_discriminator_network(
        observation_size,
        n_actions,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=discriminator_hidden_layer_sizes,
        activation=activation)
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
    )

    return GAILNetworks(discriminator_network=discriminator_network,
                        policy_network=policy_network,
                        parametric_action_distribution=parametric_action_distribution,
                        value_network=value_network)


#
# LOSS
#


def make_discriminator_loss(
        gail_networks: GAILNetworks,
        reward_scaling: float,
        discounting: float,
):
    """Creates the gail discriminator losses."""

    discriminator_network = gail_networks.discriminator_network

    def discriminator_loss(
            discriminator_params: Params,
            normalizer_params: Any,
            policy_data: dict,
            expert_data: dict,
    ) -> jnp.ndarray:

        Ds_exp = discriminator_network.apply(normalizer_params, discriminator_params, expert_data["observations"])
        D_exp = jax.vmap(lambda x, i: x.at[i].get())(Ds_exp, expert_data["action"])
        Ds_pol = discriminator_network.apply(normalizer_params, discriminator_params, policy_data["observations"])
        D_pol = jax.vmap(lambda x, i: x.at[i].get())(Ds_pol, policy_data["action"])

        eps = 1e-8
        clipped_D_exp = jnp.clip(D_exp, eps, 1 - eps)
        clipped_one_minus_D_pol = jnp.clip(1.0 - D_pol, eps, 1 - eps)

        # Binary cross entropy loss
        loss_D = -jnp.mean(jnp.log(clipped_D_exp)) - jnp.mean(jnp.log(clipped_one_minus_D_pol))
        
        return loss_D

    return discriminator_loss


def make_policy_loss(
        gail_networks: GAILNetworks,
        reward_scaling: float,
        discounting: float,
        entropy_cost: float = 1e-4,
        gae_lambda: float = 0.95,
        clipping_epsilon: float = 0.3,
        normalize_advantage: bool = True,
):
    """Creates the PPO loss for updating the gail policy."""

    policy_apply = gail_networks.policy_network.apply
    value_apply = gail_networks.value_network.apply
    discriminator_apply = gail_networks.discriminator_network.apply

    parametric_action_distribution = gail_networks.parametric_action_distribution

    def policy_loss(
            policy_and_value_params: tuple[Params, Params],
            normalizer_params: Any,
            discriminator_params: Params,
            data: Transition,
            rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, types.Metrics]:
        """Computes PPO loss."""

        policy_params, value_params = policy_and_value_params

        # first, rewrite rewards
        Ds_pol = discriminator_apply(normalizer_params, discriminator_params, data.observation)

        lookup_action = jax.vmap(jax.vmap(lambda x, i: x.at[i].get()))

        D_pol = lookup_action(Ds_pol, data.action)

        # GAIL reward: encourages (s, a) that look like expert
        eps = 1e-8
        clipped_one_minus_D_pol = jnp.clip(1.0 - D_pol, eps, 1 - eps)
        new_reward = -jnp.log(clipped_one_minus_D_pol)

        data = data._replace(reward=new_reward)

        policy_logits = policy_apply(
            normalizer_params, policy_params, data.observation
        )

        baseline = value_apply(normalizer_params, value_params, data.observation)
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        bootstrap_value = value_apply(normalizer_params, value_params, terminal_obs)

        rewards = data.reward * reward_scaling
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)

        target_action_log_probs = parametric_action_distribution.log_prob(
            policy_logits, data.extras['policy_extras']['raw_action']
        )
        behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

        vs, advantages = ppo_losses.compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=discounting,
        )
        if normalize_advantage:
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = (
            jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
        )

        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
        entropy_loss = entropy_cost * -entropy

        total_loss = policy_loss + v_loss + entropy_loss
        return total_loss

    return policy_loss


#
# TRAINING
#

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    discriminator_optimizer_state: optax.OptState
    discriminator_params: Params
    pv_optimizer_state: optax.OptState
    policy_params: Params
    value_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_map(lambda x: x[0], v)


def _init_training_state(
        key: PRNGKey,
        obs_size: int,
        local_devices_to_use: int,
        gail_networks: GAILNetworks,
        discriminator_optimizer: optax.GradientTransformation,
        pv_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_discriminator, key_policy, key_value = jax.random.split(key, 4)
    discriminator_params = gail_networks.discriminator_network.init(key_discriminator)
    discriminator_optimizer_state = discriminator_optimizer.init(discriminator_params)

    policy_params = gail_networks.policy_network.init(key_policy)
    value_params = gail_networks.value_network.init(key_value)

    pv_optimizer_state = pv_optimizer.init((policy_params, value_params))

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.float32))

    training_state = TrainingState(
        discriminator_optimizer_state=discriminator_optimizer_state,
        discriminator_params=discriminator_params,
        pv_optimizer_state=pv_optimizer_state,
        policy_params=policy_params,
        value_params=value_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        normalizer_params=normalizer_params)
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


@dataclass
class GAIL:


    discounting: float = 0.9
    expectile: float = 0.8
    reward_scaling: float = 1.
    tau: float = 0.005
    num_envs: int = 256
    num_evals: int = 16
    min_replay_size: int = 0
    max_replay_size: int = 10_000
    normalize_observations: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 256
    unroll_length: int = 64
    grad_updates_per_step: int = 1
    num_eval_envs: int = 16
    episode_length: int = 1000
    action_repeat: int = 1
    deterministic_eval: bool = True
    checkpoint_logdir: Optional[str] = None

    def train_fn(self, *, run_config, dataset, progress_fn, **_):
        """GAIL training"""

        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        device_count = local_devices_to_use * jax.process_count()

        num_evals_after_init = max(self.num_evals - 1, 1)
        num_env_steps_per_training_step = self.unroll_length * self.num_envs * self.grad_updates_per_step
        num_training_steps_per_epoch = -(
            # divide remaining steps across epochs before each evaluation
            -(run_config.num_timesteps // num_env_steps_per_training_step) // (num_evals_after_init)
        )

        #
        # ENV SET UP
        #

        training_env = dataset.get_env().wrap_for_training(episode_length=self.episode_length, action_repeat=self.action_repeat)
        env = dataset.get_eval_env(episode_length=self.episode_length, action_repeat=self.action_repeat)
        obs_size = env.obs_size

        #
        # NETWORK SET UP
        #

        normalize_fn = lambda x, y: x
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize

        network_factory = make_gail_networks
        gail_networks = network_factory(
            observation_size=obs_size,
            n_actions=env.n_actions,
            preprocess_observations_fn=normalize_fn,
            # hidden_layer_sizes=self.hidden_layer_sizes,
        )
        make_policy = get_make_policy_fn(gail_networks)

        discriminator_optimizer = optax.adam(learning_rate=self.learning_rate)
        pv_optimizer = optax.adam(learning_rate=self.learning_rate)

        # discriminator
        discriminator_loss = make_discriminator_loss(
            gail_networks,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting,
        )
        discriminator_update = gradients.gradient_update_fn(
            discriminator_loss, discriminator_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

        # policy
        policy_loss = make_policy_loss(
            gail_networks,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting,
        )
        policy_update = gradients.gradient_update_fn(
            policy_loss, pv_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)


        def sgd_step(
                carry: Tuple[TrainingState, PRNGKey],
                policy_trajectories: Transition
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry

            policy_samples_key, expert_samples_key, policy_update_key, key = jax.random.split(key, 4)

            size = jnp.array(policy_trajectories.observation.shape[:-1])
            size_len = size.shape[0]

            def random_dim_idx(key, upper_bound):
                return jax.random.randint(key, (self.batch_size,), 0, upper_bound)

            sample_keys = jax.random.split(policy_samples_key, size_len)
            full_idxs = jax.vmap(random_dim_idx, in_axes=(0, 0))(sample_keys, size)

            policy_experience = jax.tree_util.tree_map(lambda arr: arr[(*full_idxs, ...)], policy_trajectories)

            policy_data = {
                "action": policy_experience.action,
                "discount": policy_experience.discount,
                "next_observations": policy_experience.next_observation,
                "observations": policy_experience.observation,
                "reward": policy_experience.reward,
            }

            # sample expert data
            expert_data = dataset.sample(expert_samples_key, self.batch_size)

            # update normalizer params
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                jnp.concatenate((expert_data["observations"], policy_data["observations"])),
                pmap_axis_name=_PMAP_AXIS_NAME)

            d_loss, new_d_params, d_optimizer_state = discriminator_update(
                training_state.discriminator_params,
                normalizer_params,
                policy_data,
                expert_data,
                optimizer_state=training_state.discriminator_optimizer_state)

            pv_loss, (new_policy_params, new_value_params), pv_optimizer_state = policy_update(
                (training_state.policy_params, training_state.value_params),
                normalizer_params,
                new_d_params,
                policy_trajectories,
                policy_update_key,
                optimizer_state=training_state.pv_optimizer_state)

            metrics = {
                'd_loss': d_loss,
                'pv_loss': pv_loss,
            }

            new_training_state = TrainingState(
                discriminator_optimizer_state=d_optimizer_state,
                discriminator_params=new_d_params,
                pv_optimizer_state=pv_optimizer_state,
                policy_params=new_policy_params,
                value_params=new_value_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                normalizer_params=normalizer_params)
            return (new_training_state, key), metrics

        def training_step(
                training_state: TrainingState,
                env_state: envs.State,
                key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            policy_experience_key, training_key = jax.random.split(key)

            # sample policy data
            policy = make_policy(
                (training_state.normalizer_params, training_state.policy_params))

            def f(carry, unused_t):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, data = acting.generate_unroll(
                    training_env,
                    current_state,
                    policy,
                    current_key,
                    self.unroll_length,
                    extra_fields=('truncation',))
                return (next_state, next_key), data

            (env_state, _), policy_trajectories = jax.lax.scan(
                f, (env_state, policy_experience_key), (),
                length=self.grad_updates_per_step)

            training_state = training_state.replace(
                env_steps=training_state.env_steps + num_env_steps_per_training_step
            )
            
            (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), policy_trajectories)

            return training_state, env_state, metrics

        def training_epoch(
                training_state: TrainingState,
                env_state: envs.State,
                key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

            def f(carry, unused_t):
                ts, es, k = carry
                k, new_key = jax.random.split(k)
                ts, es, metrics = training_step(ts, es, k)
                return (ts, es, new_key), metrics

            (training_state, env_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, key),
                (),
                length=num_training_steps_per_epoch
            )
            metrics = jax.tree_map(jnp.mean, metrics)
            return training_state, env_state, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # Note that this is NOT a pure jittable method.
        def training_epoch_with_timing(
                training_state: TrainingState,
                env_state: envs.State,
                key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            nonlocal training_walltime
            t = time.time()
            (training_state, env_state, metrics) = training_epoch(
                training_state, env_state, key
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
            return training_state, env_state, metrics

        global_key, local_key = jax.random.split(jax.random.PRNGKey(run_config.seed))
        local_key = jax.random.fold_in(local_key, process_id)

        # Training state init
        training_state = _init_training_state(
            key=global_key,
            obs_size=obs_size,
            local_devices_to_use=local_devices_to_use,
            gail_networks=gail_networks,
            discriminator_optimizer=discriminator_optimizer,
            pv_optimizer=pv_optimizer)
        del global_key

        local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

        # Env init
        env_keys = jax.random.split(env_key, self.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.pmap(training_env.reset)(env_keys)

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
            params = _unpmap((training_state.normalizer_params, training_state.policy_params))
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
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            # Eval and logging
            if process_id == 0:
                # Save current policy.
                params = _unpmap((training_state.normalizer_params, training_state.policy_params))

                # Run evals.
                metrics = evaluator.run_evaluation(params, training_metrics)
                logging.info(metrics)
                progress_fn(current_step, metrics, params=params)

        total_steps = current_step
        assert total_steps >= run_config.num_timesteps

        params = _unpmap((training_state.normalizer_params, training_state.policy_params))

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        logging.info('total steps: %s', total_steps)
        pmap.synchronize_hosts()
        return (make_policy, params, metrics)
