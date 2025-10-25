import argparse

from imitation import algorithms
from imitation import environments


def main():
    parser = argparse.ArgumentParser(description="General training script for agents.")
    parser.add_argument('--algo', type=str, help='Name of algorithm to use for training.')
    parser.add_argument('--env', type=str, help='Name of environment to consider for training.')

    parser.add_argument('--num_timesteps', type=int, default=1000000, help='Number of timesteps allowed for training.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness in training.')
    args = parser.parse_args()

    if hasattr(algorithms, args.algo):
        AlgoClass = getattr(algorithms, args.algo)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not found")

    if hasattr(environments, args.env):
        env = getattr(environments, args.env)
    else:
        raise NotImplementedError(f"Environment {args.env} not found")


    algo = AlgoClass()

    def progress_fn(current_step, metrics):
        print(current_step)

    make_policy, params, metrics = algo.train_fn(
        num_timesteps=args.num_timesteps,
        seed=args.seed,
        env=env,
        progress_fn=progress_fn
    )


if __name__ == "__main__":
    main()
