"""Environment helpers for curiosity_a2c."""

from typing import Dict, Optional

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor


__all__ = ["make_frozenlake_env"]


def make_frozenlake_env(
    map_name: str = "8x8",
    is_slippery: bool = True,
    render_mode: Optional[str] = None,
    monitor: bool = True,
    monitor_kwargs: Optional[Dict] = None,
):
    """Create a FrozenLake environment wrapped for neural-network policies."""
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode,
        reward_schedule=(0,0,0)
    )
    env = FlattenObservation(env)
    if monitor:
        monitor_kwargs = monitor_kwargs or {}
        env = Monitor(env, **monitor_kwargs)
    return env
