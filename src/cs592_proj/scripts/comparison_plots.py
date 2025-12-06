import argparse
import json
import pathlib
from typing import Dict, Iterable

from brax.io import model
import jax
import jax.numpy as jnp


import numpy as np

from cs592_proj import algorithms
from cs592_proj import environments


from cs592_proj.algorithms.run_config import RunConfig
from cs592_proj.algorithms.bc.classifier_bc import CLASSIFIER_BC
from cs592_proj.algorithms.irl.trex import TREX
from cs592_proj.datasets.custom_dataset import CustomDataset
import cs592_proj.environments as envs  # noqa: F401 (likely used elsewhere)



def rollout_true_return(env, policy, episode_length=1000, seed=0):
    """Roll out a policy in the true environment and compute avg/std returns."""
    n_episodes = 20
    rollouts = env.rollout(
        policy, n_episodes=n_episodes, episode_length=episode_length, seed=seed
    )
    rewards = np.asarray(rollouts["reward"])
    episode_returns = rewards.sum(axis=0)
    print(episode_returns)
    reward_avg = float(np.mean(episode_returns))
    reward_std = float(np.std(episode_returns))
    return reward_avg, reward_std

def progress_logger(tag: str):
    def _log(step: int, metrics: Dict, **kwargs):
        #print(f"[{tag}] step={step} metrics={metrics}")
        print(f"[{tag}] step={step}")
    return _log


def run_training(
    dataset_paths: Iterable[pathlib.Path],
    env_name: str,
    run_config: RunConfig,
    output_dir: pathlib.Path,
):
    algos: Dict[str, object] = {
        #"TREX": TREX(),
        "CLASSIFIER_BC": CLASSIFIER_BC(num_evals=run_config.num_evals),
        # TODO: add Offline RL algorithms here, e.g. "IQL": IQL(...)
    }

    index = []  # list of metadata entries for all (dataset, algo) runs

    for ds_path in dataset_paths:
        print(f"\n=== Processing dataset: {ds_path} ===")

        # Adjust this line if you want a different loader (e.g., CustomDatasetImpl.from_npz)
        dataset = CustomDataset.from_resource_path(
            f"custom_datasets/{ds_path.name}",
            env_name=env_name,
        )

        for algo_name, algo in algos.items():
            print(f"Running {algo_name} on {ds_path.name}...")

            make_policy, params, metrics = algo.train_fn(
                run_config=run_config,
                dataset=dataset,
                progress_fn=progress_logger(f"{ds_path.name}:{algo_name}"),
            )
            env = dataset.env
            policy = make_policy(params, deterministic=True)

            # evaluate with ground-truth environment reward
            avg_reward, std_reward = rollout_true_return(env, policy, seed=0)
            print(avg_reward, std_reward)

            #out_path = output_dir / f"{algo_name}_{ds_path.stem}"
            #model.save_params(out_path, params)
            #print(f"Saved policy params to {out_path}")

            # Add one entry per (dataset, algo) to the index
            index.append(
                {
                    "dataset_name": ds_path.name,
                    #"dataset_path": str(ds_path),
                    "algo": algo_name,
                    #add avg and std of acheived reward of that policy 
                    "avg_reward": avg_reward,
                    "std_reward": std_reward,
                    #"env": env_name,
                    #"params_path": str(out_path),
                    #"num_timesteps": run_config.num_timesteps,
                    #"num_evals": run_config.num_evals,
                    #"seed": run_config.seed,
                }
            )

    # Write a single index.json that the comparison script can read later
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote index with {len(index)} entries to {index_path}")

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Train TREX and BC across datasets and save policies."
    )
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("src/cs592_proj/custom_datasets"),
        help="Directory containing .npz trajectory datasets.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="space_invaders",
        help="Environment name from cs592_proj.environments to pair with the datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("comparison_outputs"),
        help="Where to save trained policy parameter files.",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=10000,
        help="Training timesteps per run.",
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=4,
        help="Number of evals during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(args.dataset_dir.glob("*.npz"))
    if not dataset_paths:
        raise SystemExit(f"No .npz files found in {args.dataset_dir}")

    run_config = RunConfig(
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        seed=args.seed,
    )

    run_training(dataset_paths, args.env, run_config, output_dir)


if __name__ == "__main__":
    main()