"""Set up class to inject noise into demonstrations"""

from typing import Any, Dict, Optional, Union
import numpy as np
import os
import dataclasses
import gymnasium as gym
import hypothesis
import hypothesis.strategies as st
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout, types
from imitation.scripts.NTRIL.noise_injection import (
    EpsilonGreedyNoiseInjector,
    NoisyPolicy,
)
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.utils import (
    visualize_noise_levels,
    analyze_ranking_quality,
)
from imitation.algorithms import bc
from imitation.algorithms.bc import BC
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util
from imitation.util.logger import configure
from imitation.data import serialize


class DemonstrationNoiseInjection():
    """Injects noise into demonstrations via states or actions"""
    def __init__(self, noise_magnitude: float, 
                       venv: DummyVecEnv,
                       perturb_state: bool = True, 
                       perturb_action: bool = False):
        """Initialize DemonstrationNoiseInjection class.
        
        Args:
            noise_magnitude: how much to change the demonstration by
            venv: Vectorized Environment that houses demonstrations
            perturb_state: whether to perturb states
            perturb_action: whether to perturb actions
        """

        self.noise_magnitude = noise_magnitude
        self.venv = venv
        self.perturb_state = perturb_state
        self.perturb_action = perturb_action

    def perturb_demonstration(self, demonstrations: list[types.TrajectoryWithRew]) -> list[types.TrajectoryWithRew]:
        """Return a new list of (possibly) perturbed trajectories.

        This function accepts any Sequence of trajectories (including the
        HuggingFace-backed `TrajectoryDatasetSequence`) and returns a new
        regular Python `list` containing modified copies. It makes copies of
        numpy arrays before mutating to avoid read-only / shared-memory issues.
        """

        # Ensure we are working with a mutable list (the loader may return a
        # lazy, read-only sequence wrapper).
        demonstrations_list = list(demonstrations)

        obs_space = self.venv.observation_space
        act_space = self.venv.action_space

        out_trajs = []
        for demo in demonstrations_list:
            # Prepare writable copies of obs and acts. Handle DictObs
            if isinstance(demo.obs, types.DictObs):
                # copy each underlying array
                obs_unwrapped = demo.obs.unwrap()
                obs_copy_dict = {k: np.array(v, copy=True) for k, v in obs_unwrapped.items()}
                n_obs = next(iter(obs_copy_dict.values())).shape[0]
                # perturb per-timestep per-key
                for t in range(n_obs):
                    if self.perturb_state and np.random.rand() < self.noise_magnitude:
                        # try to use matching observation space for each key if available
                        try:
                            subspace = obs_space.spaces  # type: ignore[attr-defined]
                            for k in obs_copy_dict.keys():
                                if k in subspace and hasattr(subspace[k], "low"):
                                    low = subspace[k].low
                                    high = subspace[k].high
                                    obs_copy_dict[k][t] = np.random.uniform(low=low, high=high, size=obs_copy_dict[k][t].shape)
                                else:
                                    obs_copy_dict[k][t] = obs_copy_dict[k][t] + np.random.normal(scale=0.01, size=obs_copy_dict[k][t].shape)
                        except Exception:
                            # fallback: gaussian
                            for k in obs_copy_dict.keys():
                                obs_copy_dict[k][t] = obs_copy_dict[k][t] + np.random.normal(scale=0.01, size=obs_copy_dict[k][t].shape)

                new_obs = types.DictObs(obs_copy_dict)
            else:
                # numpy array observations
                obs_arr = np.array(demo.obs, copy=True)
                n_obs = obs_arr.shape[0]
                for t in range(n_obs):
                    if self.perturb_state and np.random.rand() < self.noise_magnitude:
                        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
                            obs_arr[t] = np.random.uniform(low=obs_space.low, high=obs_space.high, size=obs_arr[t].shape)
                        else:
                            obs_arr[t] = obs_arr[t] + np.random.normal(scale=0.01, size=obs_arr[t].shape)
                new_obs = obs_arr

            # Actions: iterate only over timesteps with actions
            acts_arr = np.array(demo.acts, copy=True)
            for t in range(len(acts_arr)):
                if self.perturb_action and np.random.rand() < self.noise_magnitude:
                    if hasattr(act_space, "low") and hasattr(act_space, "high"):
                        acts_arr[t] = np.random.uniform(low=act_space.low, high=act_space.high, size=acts_arr[t].shape)
                    else:
                        acts_arr[t] = acts_arr[t] + np.random.normal(scale=0.01, size=acts_arr[t].shape)

            # Create a new trajectory instance with the modified arrays
            new_demo = dataclasses.replace(demo, obs=new_obs, acts=acts_arr)
            out_trajs.append(new_demo)

        return out_trajs
    

if __name__=='__main__':
    """Perturb expert demonstrations on Mountain Car Continuous."""
    print("Generating expert demonstrations on MountainCarContinuous-v0...")

    # Path to save/load the model
    model_path = "src/imitation/scripts/NTRIL/testcode/expert_policy.zip"
    

    rngs = np.random.default_rng()
    
    # Setup environment
    venv = util.make_vec_env("MountainCarContinuous-v0", rng=rngs, post_wrappers = [lambda e, _: RolloutInfoWrapper(e)])
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        expert_policy = PPO.load(model_path, env=venv)
    else:
        # Train expert policy (or load pre-trained)
        print("Training expert policy...")
        expert_policy = PPO("MlpPolicy", venv, verbose=0)
        expert_policy.learn(total_timesteps=10000)

        # Save trained model
        expert_policy.save(model_path)

    
    # Generate expert demonstrations
    traj_path = "src/imitation/scripts/project_592/expert_demonstrations"

    if os.path.exists(traj_path):
        expert_trajectories = serialize.load(traj_path)
    else:
        expert_trajectories = rollout.rollout(
            expert_policy,
            venv,
            rollout.make_sample_until(min_episodes=10),
            rng=rngs
        )
        # Save expert trajectories
        serialize.save(traj_path, expert_trajectories)
        print(f"Nominal Expert trajectories saved to {traj_path}")
    
    print(f"Generated {len(expert_trajectories)} expert trajectories")

    # Perturb nominal demonstrations
    print("Perturbing nominal expert demonstrations...")

    noise_traj_path = "src/imitation/scripts/project_592/perturbed_demonstrations"

    noise_magnitude = 1.0
    mcc_perturber = DemonstrationNoiseInjection(noise_magnitude=noise_magnitude,
                                                venv = venv,
                                                perturb_state=True,
                                                perturb_action=False)
    

    if os.path.exists(noise_traj_path):
        perturbed_trajectories = serialize.load(noise_traj_path)
    else:
        perturbed_trajectories = mcc_perturber.perturb_demonstration(demonstrations=expert_trajectories)
        serialize.save(noise_traj_path, perturbed_trajectories)
        
    # Check for differences
    rms_differences = {'State': [], 'Action': []}

    for idx in range(len(expert_trajectories)):
        nominal_demo = expert_trajectories[idx]
        perturbed_demo = perturbed_trajectories[idx]
        
        temp_rms_obs = []
        temp_rms_acts = []

        for jdx in range(len(nominal_demo.obs)):
            obs_diff = nominal_demo.obs[jdx] - perturbed_demo.obs[jdx]
            temp_rms_obs.append(np.linalg.norm(obs_diff))
        for jdx in range(len(nominal_demo.acts)):
            acts_diff = nominal_demo.acts[jdx] - perturbed_demo.acts[jdx]
            temp_rms_acts.append(np.linalg.norm(acts_diff))

        rms_differences['State'].append(np.mean(temp_rms_obs))
        rms_differences['Action'].append(np.mean(temp_rms_acts))
    
    print(f"Average RMS deviation in states is: {np.mean(rms_differences['State'])}")
    print(f"Average RMS deviation in actions is: {np.mean(rms_differences['Action'])}")




