from dataclasses import dataclass


@dataclass
class RunConfig:

    # total number of environment timesteps to allow for training
    num_timesteps: int = 1_000_000

    # total number of evaluations to perform and log during training
    num_evals: int = 16

    # number of episodes to simulate per policy evaluation
    eval_episodes: int = 5

    # seed for randomization during training
    seed: int = 0

    eval_on_cpu: bool = True
