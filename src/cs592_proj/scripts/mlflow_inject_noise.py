import argparse
import numpy as np
import tempfile
from typing import Optional
from pathlib import Path


from brax.io import model

from cs592_proj.algorithms.run_config import RunConfig

from cs592_proj import algorithms
from cs592_proj import environments
from cs592_proj import datasets

import mlflow

def main():

    ''' Take a dataset of trajectories and inject noise to randomly disturb actions'''

    parser = argparse.ArgumentParser(description="General script for injecting noise in datasets")
    parser.add_argument('--id', type=str, help='Name of training run ID from which the policy is loaded.')
    parser.add_argument('--env', type=str, help='Name of environment to consider for noising.')
    parser.add_argument('--noise_schedule', type=list[int], default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], help='List of noise schedules representing the probability of perturbing an action in existing trajectory.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness in training.')
    parser.add_argument('--episodes_per_policy', type=int, default=100, help="Number of episodes per trained policy to collect for the dataset")
    parser.add_argument('--output', type=Path, help="Output path")
    args = parser.parse_args()

    run = mlflow.get_run(run_id=args.id)

    rng = np.random.default_rng(args.seed)
    
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

    rollouts: dict[str, np.ndarray] = env.rollout(policy, n_episodes=args.episodes_per_policy, episode_length=100, seed=0)

    # noise rollouts
    def noise_rollouts(rollouts: dict[str, np.ndarray], noise_amount) -> dict[str, np.ndarray]:

        noised_rollouts = {key: np.array(value) for key, value in rollouts.items()}
        action_list = range(env.n_actions)
        total_time = len(rollouts['observation'])[1]
        
        # iterate through each rollout in env
        for rollout_idx in range(len(args.episodes_per_policy)):

            actions = rollouts['action'][rollout_idx].copy()

            # generate flip mask
            flip_mask = rng.random(actions.shape) < noise_amount

            random_actions = rng.integers(0, env.n_actions, size=actions.shape)
            noisy_actions = np.where(flip_mask, random_actions, actions)

            # replace existing actions with new actions

            noised_rollouts['action'][rollout_idx] = noisy_actions



        return noised_rollouts
    
    # create parent directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for noise in args.noise_schedule:
        new_rollout = noise_rollouts(rollouts, noise)

        with open(args.output, "wb") as f:
            np.savez(f, **new_rollout)


    return


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/nicomiguel/CS592/local_data")
    mlflow.set_experiment("cs592-il-training")

    main()