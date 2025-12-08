import argparse
import tempfile
import numpy as np
from typing import Optional
from pathlib import Path

from enum import Enum

import jax
import jax.numpy as jnp
from brax.io import model

from gymnax.visualize import Visualizer

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
    custom = "custom" # for now, custom just chooses all runs

def main():
    parser = argparse.ArgumentParser(description="Dataset generation script for trained agents.")
    parser.add_argument('--id', type=str, help='Name of training run ID from which the policy is loaded.')
    parser.add_argument('--policies', type=PoliciesChoice.argtype, choices=PoliciesChoice, help="Way to select trained policies for collecting the dataset")
    parser.add_argument('--env', type=str, help='Name of environment to consider for training.')
    parser.add_argument('--episodes_per_policy', type=int, default=100, help="Number of episodes per trained policy to collect for the dataset")
    parser.add_argument('--output', type=Path, help="Output path")
    args = parser.parse_args()

    assert args.output.suffix == ".npz", "Output path should be an npz file"

    if args.output.exists():
        input(f"Output \"{args.output}\" exists. Continue and overwrite anyway?")

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

    artifacts = mlflow.artifacts.list_artifacts(f"runs:/{args.id}/")

    if args.policies == PoliciesChoice.best:
        # get policy parameters
        logged_model_path = f'runs:/{args.id}/policy_params'
        real_path = mlflow.artifacts.download_artifacts(logged_model_path)
        params = model.load_params(real_path)
        policies = {"best": make_policy(params)}
    elif args.policies == PoliciesChoice.custom:
        # get custom runs
        logged_model_base_path = f'runs:/{args.id}/'
        real_base_path = mlflow.artifacts.download_artifacts(logged_model_base_path)
        directory = Path(real_base_path)
        policies = {}
        for file in directory.iterdir():
            params = model.load_params(str(file))
            policies[str(file)] = make_policy(params)
    else:
        raise NotImplementedError()

    for policy_name, policy in policies.items():
        rollouts: dict[str, np.ndarray] = env.rollout(policy, n_episodes=args.episodes_per_policy, episode_length=100, seed=0)

        # Create unique filename for each policy
        if len(policies) > 1:
            output_path = args.output.parent / f"{args.output.stem}{policy_name}{args.output.suffix}"
        else:
            output_path = args.output

        # create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            np.savez(f, **rollouts)
            



if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/nicomiguel/CS592/local_data")
    mlflow.set_experiment("cs592-il-training")

    main()
