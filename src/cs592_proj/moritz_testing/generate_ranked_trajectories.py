from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

def record_video(env_name, policy, file_name=None, deterministic=True, max_steps=1000, debug=True):
    """Roll a single episode and store the video under videos/<file_name>."""
    if debug:
        print(f"record_video: starting capture for {env_name} -> {file_name or 'unnamed-policy'}")

    file_name = file_name or "unnamed-policy"
    video_dir = Path("videos") / file_name
    video_dir.mkdir(parents=True, exist_ok=True)

    env = RecordVideo(
        gym.make(env_name, render_mode="rgb_array"),
        video_folder=str(video_dir),
        name_prefix=file_name.replace(" ", "_"),
        episode_trigger=lambda _: True,
    )
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        if hasattr(policy, "predict"):
            action, _ = policy.predict(obs, deterministic=deterministic)
        else:
            action = policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    env.close()

    videos = sorted(video_dir.glob("*.mp4"))
    if videos and debug:
        print(f"record_video: saved video to {videos[-1]}")
    elif not videos and debug:
        print("record_video: no video files found")
    return videos[-1] if videos else None

def record_trajectory_video(trajectory, env_name, file_name=None, debug=True):
    """Replay a recorded trajectory and store the resulting video."""
    file_name = file_name or "trajectory"
    video_dir = Path("videos") / file_name
    video_dir.mkdir(parents=True, exist_ok=True)

    env = RecordVideo(
        gym.make(env_name, render_mode="rgb_array"),
        video_folder=str(video_dir),
        name_prefix=file_name.replace(" ", "_"),
        episode_trigger=lambda _: True,
    )

    try:
        initial_obs = getattr(trajectory, "obs", None)
        initial_obs = initial_obs[0] if initial_obs is not None else None
        env.reset()
        if initial_obs is not None and hasattr(env.unwrapped, "state"):
            try:
                if isinstance(initial_obs, dict):
                    raise TypeError('dict observations are not supported for trajectory replay')
                env.unwrapped.state = np.array(initial_obs, dtype=float).copy()
                if hasattr(env.unwrapped, "steps_beyond_terminated"):
                    env.unwrapped.steps_beyond_terminated = None
            except Exception as exc:
                if debug:
                    print(f"record_trajectory_video: failed to set env state: {exc}")

        for action in getattr(trajectory, "acts", []):
            step_action = action
            if isinstance(step_action, np.ndarray):
                if step_action.shape == ():
                    step_action = step_action.item()
                else:
                    step_action = step_action.astype(float)
            _, _, terminated, truncated, _ = env.step(step_action)
            if terminated or truncated:
                break
    finally:
        env.close()

    videos = sorted(video_dir.glob("*.mp4"))
    if videos and debug:
        print(f"record_trajectory_video: saved video to {videos[-1]}")
    elif not videos and debug:
        print("record_trajectory_video: no video files found")
    return videos[-1] if videos else None


def train_and_save_policy(iterations, env_name, file_name=None, policy=None, debug=True):
    """Train the supplied policy (or a fresh PPO policy) and persist it to disk."""
    file_name = file_name or "unnamed-policy"

    if debug:
        print(f"train_and_save_policy: training for {iterations} timesteps on {env_name} -> {file_name}")

    if policy is None:
        env = gym.make(env_name)
        model = PPO(
            policy=MlpPolicy,
            env=env,
            seed=0,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
        )
        env_to_close = env
    else:
        model = policy
        env_to_close = None
        if hasattr(model, "get_env"):
            current_env = model.get_env()
            if current_env is None:
                env = gym.make(env_name)
                model.set_env(env)
                env_to_close = env
        else:
            raise ValueError("Provided policy must expose get_env() and set_env().")

    model.learn(total_timesteps=iterations)

    save_dir = Path("policies")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file_name
    model.save(str(save_path))

    if debug:
        print(f"train_and_save_policy: saved policy to {save_path}")

    if env_to_close is not None:
        env_to_close.close()

    return model

def generate_policies_and_videos(num_policies, iteration_step, env_name):
    policy = None
    video_entries = []
    policy_entries = []

    for idx in range(1, num_policies + 1):
        total_timesteps = idx * iteration_step
        policy_filename = f"ppo_cartpole_{total_timesteps}"
        policy_path = Path("policies") / f"{policy_filename}.zip"
        video_dir = Path("videos") / policy_filename
        existing_videos = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []
        policy_entries.append(policy_path)
        if policy_path.exists() and existing_videos:
            print(f"{policy_filename} already exists.")
            policy = PPO.load(str(policy_path))
            video_entries.append((total_timesteps, existing_videos[-1]))
            continue

        policy = train_and_save_policy(
            iterations=iteration_step,
            env_name=env_name,
            file_name=policy_filename,
            policy=policy,
            debug=True,
        )
        video_path = record_video(
            env_name=env_name,
            policy=policy,
            file_name=policy_filename,
            debug=True,
        )
        video_entries.append((total_timesteps, video_path))
    return video_entries, policy_entries

def sample_trajectories_with_reward(policy, no_traj, env_name, seed=None, deterministic=True):
    """Collect `no_traj` trajectories and their returns from the given policy."""
    if no_traj <= 0:
        return []

    if hasattr(policy, "get_env") and policy.get_env() is not None:
        vec_env = policy.get_env()
        close_env = False
    else:
        vec_env = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_name))])
        close_env = True

    try:
        sample_until = rollout.make_sample_until(min_timesteps=None, min_episodes=no_traj)
        trajectories = rollout.rollout(
            policy,
            vec_env,
            sample_until,
            rng=np.random.default_rng(seed),
            deterministic_policy=deterministic,
        )
    finally:
        if close_env:
            vec_env.close()

    results = []
    for traj in trajectories[:no_traj]:
        total_reward = float(np.sum(traj.rews))
        results.append(traj)

    return results

def generate_trajectories(policy_entries, env_name, no_trajectories=3):
    trajectory_entries = []

    for policy_path in policy_entries:
        if not policy_path.exists():
            continue
        policy = PPO.load(str(policy_path))

        traj_infos = sample_trajectories_with_reward(
            policy=policy,
            no_traj=no_trajectories,
            env_name=env_name,
            deterministic=True,
        )
        
        trajectory_entries.extend(traj_infos)
    
    return trajectory_entries
