"""
Shared utilities for A2C training and testing.
"""
from typing import Optional

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from curiosity_a2c.envs import make_frozenlake_env


def make_env(render_mode: Optional[str] = None):
    """Create the default FrozenLake-v1 environment."""
    return make_frozenlake_env(
        map_name="8x8",
        is_slippery=True,
        render_mode=render_mode,
    )


def test_model(
    model_path: str,
    n_episodes: int = 10,
    model_type: str = "baseline",
    render: bool = False,
):
    """Test a trained A2C model (works for both baseline and ICM versions)."""
    model = A2C.load(model_path)

    render_mode = "human" if render else None
    env = DummyVecEnv([lambda: make_env(render_mode=render_mode)])

    try:
        vec_normalize_path = model_path.replace("_final", "") + "_vecnormalize.pkl"
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("Warning: Normalization stats not found, continuing without normalization")

    episode_rewards = []
    episode_lengths = []

    print(f"\n{'=' * 50}")
    print(f"Testing {model_type.upper()} A2C Model")
    print(f"{'=' * 50}\n")

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    env.close()

    print(f"\n{'=' * 50}")
    print(f"Test Results ({n_episodes} episodes)")
    print(f"{'=' * 50}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} (+/- {np.std(episode_rewards):.2f})")
    print(f"Average Steps: {np.mean(episode_lengths):.2f} (+/- {np.std(episode_lengths):.2f})")
    print(f"Success Rate: {sum(1 for r in episode_rewards if r > 0.0) / n_episodes * 100:.1f}%")
    print(f"{'=' * 50}\n")

    return episode_rewards, episode_lengths
