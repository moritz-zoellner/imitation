import argparse
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
    parser = argparse.ArgumentParser(description="General training script for agents.")
    parser.add_argument('--algo', type=str, help='Name of algorithm to use for training.')
    parser.add_argument('--env', type=str, help='Name of environment to consider for training.')
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training.')

    # run config arguments
    parser.add_argument('--num_timesteps', type=int, default=15000000, help='Number of timesteps allowed for training.')
    parser.add_argument('--num_evals', type=int, default=16, help='Number of evaluations to perform and log during training.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness in training.')
    args = parser.parse_args()

    print(args)

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

    def progress_fn(current_step, metrics, **kwargs):
        print(f"Logging for {current_step}")
        print(metrics)
        mlflow.log_metrics(metrics, step=current_step)

        params = kwargs["params"]

        # save params to temporary directory and then log the saved file to store in mlflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, f"policy_params_{current_step}")
            model.save_params(path, params)
            mlflow.log_artifact(path)


    run_config = RunConfig(
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        seed=args.seed
    )

    with mlflow.start_run(tags={"env": args.env, "dataset": args.dataset, "alg": args.algo}) as run:
        make_policy, params, metrics = algo.train_fn(
            run_config=run_config,
            env=env,
            dataset=dataset,
            progress_fn=progress_fn
        )

        # save params to temporary directory and then log the saved file to store in mlflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, f"policy_params")
            model.save_params(path, params)
            mlflow.log_artifact(path)


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/nicomiguel/CS592/local_data")
    mlflow.set_experiment("cs592-il-training")

    main()
