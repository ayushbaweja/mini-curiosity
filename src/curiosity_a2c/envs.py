"""Environment helpers for curiosity_a2c."""

from typing import Dict, Optional
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from .wrappers import FrozenLakePixelWrapper    
from gymnasium.wrappers import TimeLimit

__all__ = ["make_frozenlake_env"]


def make_frozenlake_env(
    map_name: str = "4x4",
    is_slippery: bool = True,
    render_mode: Optional[str] = "rgb_array",
    monitor: bool = True,
    monitor_kwargs: Optional[Dict] = None,
):
    """Create a FrozenLake environment wrapped for neural-network policies."""
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="rgb_array",
    )
    env = env.unwrapped
    env = TimeLimit(env, max_episode_steps=400)
    env = FrozenLakePixelWrapper(env)
    if monitor:
        monitor_kwargs = monitor_kwargs or {}
        env = Monitor(env, **monitor_kwargs)
    return env
