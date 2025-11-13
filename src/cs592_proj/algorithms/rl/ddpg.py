"""Deep Deterministic Policy Gradients
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

from cs592_proj.environments import jax_acting as acting
from cs592_proj.environments.jax_acting import Evaluator




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
class DDPGNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_q_network(
        obs_size: int,
        action_size: int,
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


def get_make_policy_fn(ddpg_network: DDPGNetworks, n_actions: int):

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> types.Policy:

        def random_policy(observation: types.Observation, key: PRNGKey) -> Tuple[types.Action, types.Extra]:
            action = jax.random.randint(key, (observation.shape[0],), 0, n_actions)
            return action, {}

        def greedy_policy(observation: types.Observation, key: PRNGKey) -> Tuple[types.Action, types.Extra]:
            qs = dq_network.q_network.apply(*params, observation)
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


def make_dq_networks(
        observation_size: int,
        n_actions: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: networks.ActivationFn = linen.elu) -> DQNetworks:

    q_network = make_discrete_q_network(
        observation_size,
        n_actions,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation)

    return DQNetworks(q_network=q_network)


#
# LOSS
#

def make_critic_loss(
        dq_network: DQNetworks,
        reward_scaling: float,
        discounting: float,
):
  """Creates the DQN losses."""

  q_network = dq_network.q_network

  def critic_loss(
          q_params: Params,
          normalizer_params: Any,
          target_q_params: Params,
          transitions: Transition,
          key: PRNGKey
  ) -> jnp.ndarray:

    # Q(s_t, a_t) for all actions
    qs_old = q_network.apply(normalizer_params, q_params, transitions.observation)
    q_old_action = jax.vmap(lambda x, i: x.at[i].get())(qs_old, transitions.action)

    # Q1(s_t+1, a_t+1)/Q2(s_t+1, a_t+1) for all actions
    next_double_qs = q_network.apply(normalizer_params, target_q_params, transitions.next_observation)

    # Q(s_t+1, o_t+1) for all actions
    next_qs = jnp.min(next_double_qs, axis=-1)

    # V(s_t+1) = max_o Q(s_t+1, o) (because pi is argmax Q)
    next_v = next_qs.max(axis=-1)

    # E (s_t, a_t, s_t+1) ~ D [r(s_t, a_t) + gamma * V(s_t+1)]
    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling +
                                     transitions.discount * discounting * next_v)

    # Q(s_t, a_t) - E[r(s_t, a_t) + gamma * V(s_t+1)]
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  return critic_loss

#
# TRAINING
#

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_map(lambda x: x[0], v)


def _init_training_state(
        key: PRNGKey,
        obs_size: int,
        local_devices_to_use: int,
        dq_network: DQNetworks,
        q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q, key_cost_q = jax.random.split(key, 3)
    q_params = dq_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.float32))

    training_state = TrainingState(
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        normalizer_params=normalizer_params)
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


@dataclass
class DDPG:
    """Deep Deterministic Policy Gradient
    Args:
        discounting: discount factor gamma
    """

    discounting: float = 0.9
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

    def train_fn(self, *, num_timesteps, seed, env, progress_fn):
        """Q-Learning training"""

        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        device_count = local_devices_to_use * jax.process_count()

        # environments should be allocated evenly accross devices
        assert self.num_envs % device_count == 0

        if self.min_replay_size >= num_timesteps:
            raise ValueError('No training will happen because min_replay_size >= num_timesteps')

        # each actor step steps all environments
        env_steps_per_actor_step = self.num_envs
        # number of actor steps necessary to fill replay buffer to minimum
        num_prefill_actor_steps = -(-self.min_replay_size // self.num_envs)
        # number of env steps necessary to fill replay buffer to minimum
        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step

        if num_timesteps - num_prefill_env_steps <= 0:
            raise ValueError('No training will happen because min_replay_size will exhaust num_timesteps')

        num_evals_after_init = max(self.num_evals - 1, 1)
        num_training_steps_per_epoch = -(
            # divide remaining steps across epochs before each evaluation
            -(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
        )

        #
        # ENV SET UP
        #

        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng)
        training_env = env.wrap_for_training(episode_length=self.episode_length, action_repeat=self.action_repeat)

        obs_size = env.obs_size

        #
        # Q-NETWORK SET UP
        #

        normalize_fn = lambda x, y: x
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize

        network_factory = make_dq_networks
        dq_network = network_factory(
            observation_size=obs_size,
            n_actions=env.n_actions,
            preprocess_observations_fn=normalize_fn,
            hidden_layer_sizes=self.hidden_layer_sizes,
        )
        make_policy = get_make_policy_fn(dq_network, n_actions=env.n_actions)

        q_optimizer = optax.adam(learning_rate=self.learning_rate)

        dummy_obs = jnp.zeros((obs_size,))
        dummy_transition = Transition(
            observation=dummy_obs,
            action=0,
            reward=0.,
            discount=0.,
            next_observation=dummy_obs,
            extras={
                'state_extras': {'truncation': 0.},
                'policy_extras': {},
            }
        )
        replay_buffer = replay_buffers.UniformSamplingQueue(
            max_replay_size=self.max_replay_size // device_count,
            dummy_data_sample=dummy_transition,
            sample_batch_size=self.batch_size * self.grad_updates_per_step // device_count)

        critic_loss = make_critic_loss(
            dq_network=dq_network,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting,
        )
        critic_update = gradients.gradient_update_fn(
            critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

        def sgd_step(
            carry: Tuple[TrainingState, PRNGKey],
            transitions: Transition
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry

            key, key_critic = jax.random.split(key, 2)

            critic_loss, q_params, q_optimizer_state = critic_update(
                training_state.q_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state)

            new_target_q_params = jax.tree_map(lambda x, y: x * (1 - self.tau) + y * self.tau,
                                               training_state.target_q_params, q_params)

            metrics = {
                'critic_loss': critic_loss,
            }

            new_training_state = TrainingState(
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                normalizer_params=training_state.normalizer_params)
            return (new_training_state, key), metrics


        def get_experience(
                normalizer_params: running_statistics.RunningStatisticsState,
                q_params: Params,
                env_state: envs.State,
                buffer_state: ReplayBufferState,
                key: PRNGKey
        ) -> Tuple[
            running_statistics.RunningStatisticsState,
            envs.State,
            ReplayBufferState
        ]:
            policy = make_policy((normalizer_params, q_params))
            env_state, transitions = acting.actor_step(
                training_env,
                env_state,
                policy,
                key,
                extra_fields=(
                    'truncation',
                )
            )

            normalizer_params = running_statistics.update(
                normalizer_params,
                transitions.observation,
                pmap_axis_name=_PMAP_AXIS_NAME)

            buffer_state = replay_buffer.insert(buffer_state, transitions)
            return normalizer_params, env_state, buffer_state

        def training_step(
                training_state: TrainingState,
                env_state: envs.State,
                buffer_state: ReplayBufferState,
                key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            experience_key, training_key = jax.random.split(key)
            normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.q_params,
                env_state,
                buffer_state,
                experience_key
            )
            training_state = training_state.replace(normalizer_params=normalizer_params, env_steps=training_state.env_steps + env_steps_per_actor_step)
            
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            # Change the front dimension of transitions so 'update_step' is called
            # grad_updates_per_step times by the scan.
            transitions = jax.tree_map(lambda x: jnp.reshape(x, (self.grad_updates_per_step, -1) + x.shape[1:]), transitions)
            (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
            
            metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        def prefill_replay_buffer(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                new_normalizer_params, env_state, buffer_state = get_experience(
                    training_state.normalizer_params,
                    training_state.q_params,
                    env_state,
                    buffer_state,
                    key,
                )
                new_training_state = training_state.replace(
                    normalizer_params=new_normalizer_params,
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (new_training_state, env_state, buffer_state, new_key), ()
  
            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps
            )[0]

        prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

        def training_epoch(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

          def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

          (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
              f,
              (training_state, env_state, buffer_state, key),
              (),
              length=num_training_steps_per_epoch
          )
          metrics = jax.tree_map(jnp.mean, metrics)
          return training_state, env_state, buffer_state, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # Note that this is NOT a pure jittable method.
        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
          nonlocal training_walltime
          t = time.time()
          (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
          )
          metrics = jax.tree_map(jnp.mean, metrics)
          jax.tree_map(lambda x: x.block_until_ready(), metrics)

          epoch_training_time = time.time() - t
          training_walltime += epoch_training_time
          sps = (env_steps_per_actor_step *
                 num_training_steps_per_epoch) / epoch_training_time
          metrics = {
              'training/sps': sps,
              'training/walltime': training_walltime,
              **{f'training/{name}': value for name, value in metrics.items()}
          }
          return training_state, env_state, buffer_state, metrics

        global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
        local_key = jax.random.fold_in(local_key, process_id)

        # Training state init
        training_state = _init_training_state(
            key=global_key,
            obs_size=obs_size,
            local_devices_to_use=local_devices_to_use,
            dq_network=dq_network,
            q_optimizer=q_optimizer)
        del global_key

        local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

        # Env init
        env_keys = jax.random.split(env_key, self.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.pmap(training_env.reset)(env_keys)

        # Replay buffer init
        buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

        eval_env = env.wrap_for_eval(episode_length=self.episode_length, action_repeat=self.action_repeat)
        evaluator = Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key=eval_key
        )

        # Run initial eval
        if process_id == 0 and self.num_evals > 1:
          metrics = evaluator.run_evaluation(
              _unpmap((training_state.normalizer_params, training_state.q_params)),
              training_metrics={})
          logging.info(metrics)
          progress_fn(0, metrics)

        # Create and initialize the replay buffer.
        t = time.time()
        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_keys
        )

        replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
        logging.info('replay size after prefill %s', replay_size)
        assert replay_size >= self.min_replay_size
        training_walltime = time.time() - t

        current_step = 0
        for _ in range(num_evals_after_init):
          logging.info('step %s', current_step)

          # Optimization
          epoch_key, local_key = jax.random.split(local_key)
          epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
          (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
          )
          current_step = int(_unpmap(training_state.env_steps))

          # Eval and logging
          if process_id == 0:
            if self.checkpoint_logdir:
              # Save current policy.
              params = _unpmap((training_state.normalizer_params, training_state.q_params))
              path = f'{self.checkpoint_logdir}_dq_{current_step}.pkl'
              model.save_params(path, params)

            # Run evals.
            metrics = evaluator.run_evaluation(_unpmap((training_state.normalizer_params, training_state.q_params)), training_metrics)
            logging.info(metrics)
            progress_fn(current_step, metrics)

        total_steps = current_step
        assert total_steps >= num_timesteps

        params = _unpmap((training_state.normalizer_params, training_state.q_params))

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        logging.info('total steps: %s', total_steps)
        pmap.synchronize_hosts()
        return (make_policy, params, metrics)
