"""Behavior Cloning with a classifier
"""

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Sequence, Union

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
from distrax import Softmax
from optax.losses import safe_softmax_cross_entropy

from cs592_proj.environments import jax_acting as acting
from cs592_proj.environments.jax_acting import Evaluator


Params = Any
Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'

#
# DISTRIBUTION
#


# https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/bijectors/identity.py
class IdentityBijector:
    """Identity Bijector."""

    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def forward_log_det_jacobian(self, x):
        return 0 # tf.constant(0., dtype=x.dtype)


class SoftmaxDistribution(distribution.ParametricDistribution):
    """Softmax distribution"""

    def __init__(self, event_size, min_std=0.001, var_scale=1):
        """Initialize the distribution.
    
        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.
          var_scale: adjust the gaussian's scale parameter.
        """
        super().__init__(
            param_size=event_size,
            postprocessor=IdentityBijector(),
            event_ndims=0,
            reparametrizable=True
        )
        self._temperature = 1

    def create_dist(self, parameters):
        return Softmax(logits=parameters, temperature=self._temperature)


#
# Q NETWORK IMPLEMENTATION
#

@flax.struct.dataclass
class Classifier:
    classifier_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_classifier_network(
        obs_size: int,
        n_actions: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a classifier for discrete action spaces."""

    # Q function for discrete space of actions.
    class ClassifierModule(linen.Module):

        @linen.compact
        def __call__(self, obs: jnp.ndarray):
            z = MLP(
                layer_sizes=list(hidden_layer_sizes) + [n_actions],
                activation=activation,
                kernel_init=jax.nn.initializers.lecun_uniform()
            )(obs)
            log_action_probs = linen.activation.log_softmax(z)
            return log_action_probs

    classifier_module = ClassifierModule()

    def apply(processor_params, classifier_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return classifier_module.apply(classifier_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: classifier_module.init(key, dummy_obs), apply=apply)


def get_make_policy_fn(classifier: Classifier):

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> types.Policy:

        def policy(observation: types.Observation,
                   key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = classifier.classifier_network.apply(*params, observation)
            if deterministic:
                return classifier.parametric_action_distribution.mode(logits), {}
            origin_action = classifier.parametric_action_distribution.sample_no_postprocessing(logits, key_sample)
            return classifier.parametric_action_distribution.postprocess(origin_action), {}

        return policy

    return make_policy


def make_classifier(
        observation_size: int,
        n_actions: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: networks.ActivationFn = linen.elu) -> Classifier:

    classifier_network = make_classifier_network(
        observation_size,
        n_actions,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation)

    parametric_action_distribution = SoftmaxDistribution(event_size=n_actions)

    return Classifier(
        classifier_network=classifier_network,
        parametric_action_distribution=parametric_action_distribution)


#
# LOSS
#

def make_classifier_loss(classifier: Classifier, n_actions: int):
    """Creates the classifier losses."""

    classifier_network = classifier.classifier_network

    def critic_loss(
            classifier_params: Params,
            normalizer_params: Any,
            data: dict[str, jax.Array],
            key: PRNGKey
    ) -> jnp.ndarray:
        predictions = classifier_network.apply(normalizer_params, classifier_params, data["observations"])
        labels = jax.nn.one_hot(data["action"], n_actions)
        loss = safe_softmax_cross_entropy(predictions, labels)
        return loss.mean()

    return critic_loss


#
# TRAINING
#

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    classifier_optimizer_state: optax.OptState
    classifier_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_map(lambda x: x[0], v)


def _init_training_state(
        key: PRNGKey,
        obs_size: int,
        local_devices_to_use: int,
        classifier: Classifier,
        classifier_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    classifier_params = classifier.classifier_network.init(key)
    classifier_optimizer_state = classifier_optimizer.init(classifier_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.float32))

    training_state = TrainingState(
        classifier_optimizer_state=classifier_optimizer_state,
        classifier_params=classifier_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        normalizer_params=normalizer_params)
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


@dataclass
class CLASSIFIER_BC:
    """Behavior cloning with a discrete action space using a classifier
    """

    num_evals: int = 16
    # min_replay_size: int = 0
    # max_replay_size: int = 10_000
    normalize_observations: bool = True
    hidden_layer_sizes: Sequence[int] = (256, 256)
    learning_rate: float = 1e-4
    batch_size: int = 256
    grad_updates_per_step: int = 1
    num_eval_envs: int = 16
    episode_length: int = 1000
    action_repeat: int = 1
    deterministic_eval: bool = True
    # checkpoint_logdir: Optional[str] = None

    def get_make_policy_fn(self, *, dataset, **_):
        env = dataset.get_env()
        training_env = env.wrap_for_training(episode_length=self.episode_length, action_repeat=self.action_repeat)
        obs_size = training_env.obs_size

        normalize_fn = lambda x, y: x
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize

        network_factory = make_classifier
        classifier = network_factory(
            observation_size=obs_size,
            n_actions=env.n_actions,
            preprocess_observations_fn=normalize_fn,
            hidden_layer_sizes=self.hidden_layer_sizes,
        )
        make_policy = get_make_policy_fn(classifier)
        return make_policy

    def train_fn(self, *, run_config, dataset, progress_fn, **_):
        """Classifier BC training"""

        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        device_count = local_devices_to_use * jax.process_count()

        num_evals_after_init = max(self.num_evals - 1, 1)
        num_training_steps_per_epoch = -(
            # divide remaining steps across epochs before each evaluation
            -(run_config.num_timesteps // self.batch_size) // (num_evals_after_init)
        )

        # #
        # # ENV SET UP
        # #

        # rng = jax.random.PRNGKey(run_config.seed)
        # rng, key = jax.random.split(rng)
        # training_env = env.wrap_for_training(episode_length=self.episode_length, action_repeat=self.action_repeat)

        env = dataset.get_eval_env(episode_length=self.episode_length, action_repeat=self.action_repeat)
        obs_size = env.obs_size

        #
        # POLICY NETWORK SET UP
        #

        normalize_fn = lambda x, y: x
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize

        network_factory = make_classifier
        classifier = network_factory(
            observation_size=obs_size,
            n_actions=env.n_actions,
            preprocess_observations_fn=normalize_fn,
            hidden_layer_sizes=self.hidden_layer_sizes,
        )
        make_policy = get_make_policy_fn(classifier)

        classifier_optimizer = optax.adam(learning_rate=self.learning_rate)

        classifier_loss = make_classifier_loss(classifier, n_actions=env.n_actions)
        classifier_update = gradients.gradient_update_fn(
            classifier_loss, classifier_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

        def sgd_step(
            carry: Tuple[TrainingState, PRNGKey],
            data: dict[str, jax.Array]
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry

            key, key_critic = jax.random.split(key, 2)

            classifier_loss, classifier_params, classifier_optimizer_state = classifier_update(
                training_state.classifier_params,
                training_state.normalizer_params,
                data,
                key_critic,
                optimizer_state=training_state.classifier_optimizer_state)

            metrics = {
                'classifier_loss': classifier_loss,
            }

            new_training_state = TrainingState(
                classifier_optimizer_state=classifier_optimizer_state,
                classifier_params=classifier_params,
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
            classifier=classifier,
            classifier_optimizer=classifier_optimizer)
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
            params = _unpmap((training_state.normalizer_params, training_state.classifier_params))
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
                params = _unpmap((training_state.normalizer_params, training_state.classifier_params))

                # Run evals.
                metrics = evaluator.run_evaluation(params, training_metrics)
                logging.info(metrics)
                progress_fn(current_step, metrics, params=params)

        total_steps = current_step
        # In practice the ceil-based epoch calculation can leave us a bit short
        # of the requested timesteps. Rather than crash, emit a warning so the
        # caller can still use the policy.
        if total_steps < run_config.num_timesteps:
            logging.warning(
                "BC finished with total_steps=%s < requested %s; continuing anyway.",
                total_steps,
                run_config.num_timesteps,
            )

        params = _unpmap((training_state.normalizer_params, training_state.classifier_params))

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        logging.info('total steps: %s', total_steps)
        pmap.synchronize_hosts()
        return (make_policy, params, metrics)


# import jax.numpy as jnp
# import jax
# from jax import random

# from flax import linen as nn
# from einops import rearrange
# import optax
# import tensorflow as tf
# from tqdm import trange
# import matplotlib.pyplot as plt


# class MLP(nn.Module):
#     """ A simple MLP, used for the encoder and decoder.
#     """
#     hidden_dim: int = 256
#     out_dim: int = 2
#     n_layers: int = 4

#     @nn.compact
#     def __call__(self, x):
#         for _ in range(self.n_layers):
#             x = nn.Dense(features=self.hidden_dim)(x)
#             x = nn.gelu(x)
#         x = nn.Dense(features=self.out_dim)(x)
#         return x

# class VAE(nn.Module):
#     """ A simple variational auto-encoder module.
#     """
#     num_latents: int = 4
#     num_out: int = 2

#     def setup(self):
#         self.encoder = MLP(out_dim=self.num_latents * 2)
#         self.decoder = MLP(out_dim=28 * 28)

#     def __call__(self, x, beta, z_rng):

#         # Flatten x
#         x = rearrange(x, 'b h w c -> b (h w c)')

#         # Concatenate x and beta
#         x = jnp.concatenate([x, beta], axis=-1)

#         # Get variational parameters from encoder
#         enc = self.encoder(x)  # Shape (batch_size, num_latents * 2)
#         enc = rearrange(enc, 'b (n c) -> b n c', c=2)  # Reshape to (batch_size, num_latents, 2)
#         mu, logvar = enc[:, :, 0], enc[:, :, 1]

#         # Sample from variational distrib. of latents
#         z = tfp.distributions.Normal(loc=mu, scale=jnp.exp(0.5 * logvar)).sample(seed=z_rng)

#         # Concatenate z and beta
#         z = jnp.concatenate([z, beta], axis=-1)

#         # Decode
#         recon_x = self.decoder(z)

#         return recon_x, mu, logvar

# @jax.vmap
# def rate(mu, logvar):
#     """ KL-divergence between latent variational distribution and unit Normal prior.
#     """
#     prior_latent = tfp.distributions.Normal(loc=0., scale=1.)  # Prior
#     q_latent = tfp.distributions.Normal(loc=mu, scale=jnp.exp(0.5 * logvar))  # Variational latent distrib.

#     return tfp.distributions.kl_divergence(q_latent, prior_latent)

# @jax.vmap
# def distortion(pred, true, beta):
#     """ "Reconstruction" loss, Gaussian noise model.
#     """
#     true = rearrange(true, 'h w c -> (h w c)')
#     # Restrict between 0 and 1 with sigmoid
#     pred = nn.sigmoid(pred)
#     log_prob = tfp.distributions.Normal(loc=pred, scale=beta).unnormalized_log_prob(true)
#     return -log_prob

# from functools import partial

# @partial(jax.jit, static_argnums=(1,))
# def loss_fn(params, vae, x_batch, log_beta_batch, z_rng):
#     """ Loss function for the VAE, rate + distortion.
#     """

#     beta_batch = jnp.power(10., log_beta_batch)

#     recon_x, mean, logvar = vae.apply(params, x_batch, beta_batch, z_rng)

#     R = rate(mean, logvar).mean(-1)
#     D = distortion(recon_x, x_batch, beta_batch).mean(-1)

#     loss = D + R
#     return loss.mean()

# num_latents = 64
# num_out = 28 * 28

# vae = VAE(num_latents=num_latents, num_out=num_out)
# key = jax.random.PRNGKey(42)
# key, z_key = random.split(key)
# _, params = vae.init_with_output(key, x[:16], jnp.ones((16, 1)), z_key)
