import argparse
import numpy as np
import tempfile
from typing import Optional
from pathlib import Path
from enum import Enum


from brax.io import model

from cs592_proj.algorithms.run_config import RunConfig

from cs592_proj import algorithms
from cs592_proj import environments
from cs592_proj import datasets

import mlflow

class ArgTypeMixin(Enum):
    @classmethod
    def argtype(cls, s: str) -> Enum:
        try:
            return cls[s]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"{s!r} is not a valid {cls.__name__}")

    def __str__(self):
        return self.name


class PoliciesChoice(ArgTypeMixin, Enum):
    best = "best"
    custom = "custom"  # chooses all policy checkpoints

def main():

    ''' Take a dataset of trajectories and inject noise to randomly disturb actions'''

    parser = argparse.ArgumentParser(description="General script for injecting noise in datasets")
    parser.add_argument('--id', type=str, help='Name of training run ID from which the policy is loaded.')
    parser.add_argument('--policies', type=PoliciesChoice.argtype, choices=PoliciesChoice, help="Way to select trained policies for collecting the dataset")
    parser.add_argument('--env', type=str, help='Name of environment to consider for noising.')
    parser.add_argument('--noise_schedule', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], help='List of noise schedules representing the probability of perturbing an action in existing trajectory.')
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

    # Load policies based on selection method (from mlflow_generate_dataset.py)
    if args.policies == PoliciesChoice.best:
        # get policy parameters
        logged_model_path = f'runs:/{args.id}/policy_params'
        real_path = mlflow.artifacts.download_artifacts(logged_model_path)
        params = model.load_params(real_path)
        policies = {"best": make_policy(params)}
    elif args.policies == PoliciesChoice.custom:
        # get custom runs - load all checkpoints
        logged_model_base_path = f'runs:/{args.id}/'
        real_base_path = mlflow.artifacts.download_artifacts(logged_model_base_path)
        directory = Path(real_base_path)
        policies = {}
        for file in directory.iterdir():
            if file.is_file() and file.name.startswith('policy_params'):
                params = model.load_params(str(file))
                policies[file.name] = make_policy(params)
        
        if not policies:
            raise ValueError(f"No policy files found in {directory}")
    else:
        raise NotImplementedError()

    # make policy
    policy = make_policy(params)

    rollouts: dict[str, np.ndarray] = env.rollout(policy, n_episodes=args.episodes_per_policy, episode_length=100, seed=0)

    # noise rollouts
    def noise_rollouts(rollouts: dict[str, np.ndarray], noise_amount) -> dict[str, np.ndarray]:

        noised_rollouts = {key: np.array(value) for key, value in rollouts.items()}
        
        # iterate through each rollout in env
        for rollout_idx in range(args.episodes_per_policy):

            actions = rollouts['action'][rollout_idx].copy()

            # generate flip mask
            flip_mask = rng.random(actions.shape) < float(noise_amount)

            random_actions = rng.integers(0, env.n_actions, size=actions.shape)
            noisy_actions = np.where(flip_mask, random_actions, actions)

            # replace existing actions with new actions

            noised_rollouts['action'][rollout_idx] = noisy_actions

        return noised_rollouts
    
    # Iterate through each policy
    for policy_name, policy in policies.items():
        print(f"Generating rollouts for policy: {policy_name}")
        
        # Generate clean rollouts
        rollouts: dict[str, np.ndarray] = env.rollout(
            policy, 
            n_episodes=args.episodes_per_policy, 
            episode_length=100, 
            seed=args.seed
        )
        
        # Apply each noise level
        for noise_level in args.noise_schedule:
            print(f"  Applying noise level: {noise_level}")
            
            # Apply noise
            noised_rollouts = noise_rollouts(rollouts, noise_level)
            
            # Create unique filename for each policy and noise level
            if len(policies) > 1 and len(args.noise_schedule) > 1:
                # Multiple policies and noise levels
                output_path = args.output.parent / f"{args.output.stem}_{policy_name}_noise{noise_level}{args.output.suffix}"
            elif len(policies) > 1:
                # Multiple policies, single noise level
                output_path = args.output.parent / f"{args.output.stem}_{policy_name}{args.output.suffix}"
            elif len(args.noise_schedule) > 1:
                # Single policy, multiple noise levels
                output_path = args.output.parent / f"{args.output.stem}_noise{noise_level}{args.output.suffix}"
            else:
                # Single policy, single noise level
                output_path = args.output
            
            # Create parent directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save noised rollouts
            with open(output_path, "wb") as f:
                np.savez(f, **noised_rollouts)
            
            print(f"Saved to: {output_path}")


    return


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/nicomiguel/CS592/local_data")
    mlflow.set_experiment("cs592-il-training")

    main()