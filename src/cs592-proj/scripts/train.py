from typing import Optional
import argparse

from imitation.algorithms.run_config import RunConfig

from imitation import algorithms
from imitation import environments
from imitation import datasets


def main():
    parser = argparse.ArgumentParser(description="General training script for agents.")
    parser.add_argument('--algo', type=str, help='Name of algorithm to use for training.')
    parser.add_argument('--env', type=str, help='Name of environment to consider for training.')
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training.')

    # run config arguments
    parser.add_argument('--num_timesteps', type=int, default=1000000, help='Number of timesteps allowed for training.')
    parser.add_argument('--num_evals', type=int, default=16, help='Number of evaluations to perform and log during training.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness in training.')
    args = parser.parse_args()

    if hasattr(algorithms, args.algo):
        AlgoClass = getattr(algorithms, args.algo)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not found")

    if args.env:
        if hasattr(environments, args.env):
            env = getattr(environments, args.env)
        else:
            raise NotImplementedError(f"Environment {args.env} not found")
    else:
        env = None

    if args.dataset:
        if hasattr(datasets, args.dataset):
            dataset = getattr(datasets, args.dataset)
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not found")
    else:
        dataset = None

    algo = AlgoClass()

    def progress_fn(current_step, metrics):
        print(current_step)

    run_config = RunConfig(
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        seed=args.seed
    )

    make_policy, params, metrics = algo.train_fn(
        run_config=run_config,
        env=env,
        dataset=dataset,
        progress_fn=progress_fn
    )


if __name__ == "__main__":
    main()
