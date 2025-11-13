import argparse
import tempfile
from typing import Optional
from pathlib import Path

import jax
import jax.numpy as jnp
from brax.io import model

from gymnax.visualize import Visualizer

from imitation.algorithms.run_config import RunConfig

from imitation import algorithms
from imitation import environments
from imitation import datasets

import mlflow


def main():
    parser = argparse.ArgumentParser(description="General training script for agents.")
    parser.add_argument('--id', type=str, help='Name of training run ID from which the policy is loaded.')
    args = parser.parse_args()

    run = mlflow.get_run(run_id=args.id)

    algo = run.data.tags["alg"]
    dataset = run.data.tags["dataset"]
    env = run.data.tags["env"]
    
    if hasattr(algorithms, algo):
        AlgoClass = getattr(algorithms, algo)
    else:
        raise NotImplementedError(f"Algorithm {algo} not found")

    if env:
        if hasattr(environments, env):
            env = getattr(environments, env)
        else:
            raise NotImplementedError(f"Environment {env} not found")
    else:
        env = None

    if dataset:
        if hasattr(datasets, dataset):
            dataset = getattr(datasets, dataset)
        else:
            raise NotImplementedError(f"Dataset {dataset} not found")
    else:
        dataset = None

    algo = AlgoClass()

    # get make_policy function
    make_policy = algo.get_make_policy_fn(env=env, dataset=dataset)

    # get policy parameters
    logged_model_path = f'runs:/{args.id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)

    # make policy
    policy = make_policy(params)

    env = env.wrap_for_visualization()

    # visualize policy (TODO: abstract)
    state_seq, reward_seq = [], []
    key = jax.random.PRNGKey(0)
    key, key_reset = jax.random.split(key)
    state = env.reset(key_reset)
    i = 0
    while i < 300:
        state_seq.append(state.state_impl)
        key, key_act, key_step = jax.random.split(key, 3)
        obs = jnp.expand_dims(state.obs, 0)        
        action, _ = policy(obs, key_act)
        state = env.step(key_step, state, action.squeeze())
        reward_seq.append(state.reward)
        if state.done:
            break
        i += 1

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env.env_impl, env.env_params, state_seq, cum_rewards)
    vis.animate(f"docs/anim.gif")

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("cs592-il-training")

    main()


# from gymnax.visualize import Visualizer

# state_seq, reward_seq = [], []
# key, key_reset = jax.random.split(key)
# obs, env_state = env.reset(key_reset, env_params)
# while True:
#     state_seq.append(env_state)
#     key, key_act, key_step = jax.random.split(key, 3)
#     action = env.action_space(env_params).sample(key_act)
#     next_obs, next_env_state, reward, done, info = env.step(
#         key_step, env_state, action, env_params
#     )
#     reward_seq.append(reward)
#     if done:
#         break
#     else:
#       obs = next_obs
#       env_state = next_env_state

# cum_rewards = jnp.cumsum(jnp.array(reward_seq))
# vis = Visualizer(env, env_params, state_seq, cum_rewards)
# vis.animate(f"docs/anim.gif")
