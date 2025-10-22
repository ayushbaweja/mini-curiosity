"""Video recording helpers for the FrozenLake baseline."""

from pathlib import Path
import sys

if __package__ in (None, ''):
    # Running as a script; ensure project src/ is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from curiosity_a2c.envs import make_frozenlake_env


def _build_env_factory(
    record: bool,
    video_folder: Optional[str],
    name_prefix: Optional[str],
    fps: int,
) -> Callable[[], gym.Env]:
    """Return a callable that creates a (possibly recording) FrozenLake env."""

    def _init():
        env = make_frozenlake_env(
            render_mode='rgb_array' if record else None,
            monitor=False,
        )
        if record:
            if video_folder is None or name_prefix is None:
                raise ValueError("Recording requires a video_folder and name_prefix.")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda _: True,
                name_prefix=name_prefix,
                metadata={'fps': fps},
            )
        return env

    return _init


def record_episodes(
    model_path: str = "a2c_frozenlake_final",
    episode_numbers=None,
    video_folder: str = "videos",
    fps: int = 30,
):
    """Record specific FrozenLake episodes and return statistics."""
    episode_numbers = episode_numbers or [1, 5, 10]
    os.makedirs(video_folder, exist_ok=True)

    print(f"Loading model from {model_path}.zip...")
    model = A2C.load(model_path)

    episode_stats = []
    videos_recorded = []
    max_episode = max(episode_numbers)

    for episode_num in range(1, max_episode + 1):
        is_recorded = episode_num in episode_numbers
        if is_recorded:
            print(f"\n📹 Recording Episode {episode_num}...")

        env = DummyVecEnv([
            _build_env_factory(
                record=is_recorded,
                video_folder=video_folder,
                name_prefix=f"episode_{episode_num}",
                fps=fps,
            )
        ])

        try:
            try:
                env = VecNormalize.load(
                    f"{model_path.replace('_final', '')}_vecnormalize.pkl",
                    env,
                )
                env.training = False
                env.norm_reward = False
            except FileNotFoundError:
                print("Warning: Normalization stats not found")

            obs = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                steps += 1

            episode_stats.append({
                'episode': episode_num,
                'reward': episode_reward,
                'steps': steps,
            })

            if is_recorded:
                print(
                    f"✅ Episode {episode_num}: Reward = {episode_reward:.2f}, Steps = {steps}"
                )
                videos_recorded.append(episode_num)
        finally:
            env.close()

    print(f"\n{'=' * 60}")
    print("Video Recording Complete!")
    print(f"{'=' * 60}")
    print(f"Videos saved to: ./{video_folder}/")
    print(f"Episodes recorded: {videos_recorded}")
    print("\nAll episode statistics:")
    for stat in episode_stats:
        marker = "📹" if stat['episode'] in episode_numbers else "  "
        print(
            f"{marker} Episode {stat['episode']}: "
            f"Reward = {stat['reward']:.2f}, Steps = {stat['steps']}"
        )
    print(f"{'=' * 60}\n")

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    print("Video files created:")
    for vf in sorted(video_files):
        print(f"  - {video_folder}/{vf}")

    return episode_stats


def record_single_episode(
    model_path: str = "a2c_frozenlake_final",
    episode_name: str = "test",
    video_folder: str = "videos",
    fps: int = 30,
):
    """Record a single FrozenLake episode with a custom filename."""
    os.makedirs(video_folder, exist_ok=True)
    model = A2C.load(model_path)

    env = DummyVecEnv([
        _build_env_factory(
            record=True,
            video_folder=video_folder,
            name_prefix=episode_name,
            fps=fps,
        )
    ])

    try:
        try:
            env = VecNormalize.load(
                f"{model_path.replace('_final', '')}_vecnormalize.pkl",
                env,
            )
            env.training = False
            env.norm_reward = False
        except FileNotFoundError:
            pass

        print(f"📹 Recording {episode_name}...")
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
    finally:
        env.close()

    print(f"✅ Video saved: {video_folder}/{episode_name}-episode-0.mp4")
    print(f"   Reward: {episode_reward:.2f}, Steps: {steps}")

    return episode_reward, steps


if __name__ == "__main__":
    import sys

    episodes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 5, 10]
    if sys.argv[1:]:
        print(f"Recording episodes: {episodes}")
    else:
        print(f"Recording default episodes: {episodes}")

    record_episodes(
        model_path="a2c_frozenlake_final",
        episode_numbers=episodes,
        video_folder="videos",
    )

    print("\n💡 Tip: You can specify episodes like:")
    print("   python record_videos.py 1 5 10")
    print("   python record_videos.py 3 7 15 20")
