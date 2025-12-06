"""Video recording utilities for FrozenLake A2C agents."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    # Running as a script; ensure project src/ is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
from typing import Callable, Optional, Sequence

import numpy as np
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
        # FIX: Explicitly set map_name="4x4" and is_slippery=True to match 
        # the configuration used during training (16 states).
        env = make_frozenlake_env(
            map_name="4x4",
            is_slippery=True,
            render_mode='rgb_array' if record else None,
            monitor=False,
        )
        if record:
            if video_folder is None or name_prefix is None:
                raise ValueError("Recording requires a video_folder and name_prefix.")
            # Gymnasium 0.29 removed the ``metadata`` keyword from RecordVideo
            # but still relies on the env's metadata to derive FPS. Guard against
            # missing metadata so video playback isn't absurdly slow/fast.
            metadata = dict(getattr(env, 'metadata', {}) or {})
            metadata['render_fps'] = fps
            env.metadata = metadata
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda _: True,
                name_prefix=name_prefix,
                video_length=0,
            )
        return env

    return _init


def record_episodes(
    model_path: str = "models/baseline/a2c_frozenlake_baseline_final",
    episode_numbers: Optional[Sequence[int]] = None,
    video_folder: str = "videos",
    fps: int = 30,
    model_type: str = 'baseline',
):
    """Record specific FrozenLake episodes and return statistics."""
    episode_numbers = list(episode_numbers) if episode_numbers is not None else [1, 5, 10]
    os.makedirs(video_folder, exist_ok=True)

    print(f"Loading {model_type.upper()} model from {model_path}.zip...")
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
                name_prefix=f"{model_type}_episode_{episode_num}",
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
                    f"✅ Episode {episode_num}: Reward = {episode_reward:.2f}, Steps: {steps}"
                )
                videos_recorded.append(episode_num)
        finally:
            env.close()

    print(f"\n{'=' * 60}")
    print(f"Video Recording Complete! ({model_type.upper()})")
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
    model_path: str = "models/baseline/a2c_frozenlake_baseline_final",
    episode_name: str = "test",
    video_folder: str = "videos",
    fps: int = 30,
    model_type: str = 'baseline',
):
    """Record a single FrozenLake episode with a custom filename."""
    os.makedirs(video_folder, exist_ok=True)
    model = A2C.load(model_path)

    env = DummyVecEnv([
        _build_env_factory(
            record=True,
            video_folder=video_folder,
            name_prefix=f"{model_type}_{episode_name}",
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

        print(f"📹 Recording {model_type.upper()} - {episode_name}...")
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

    print(f"✅ Video saved: {video_folder}/{model_type}_{episode_name}-episode-0.mp4")
    print(f"   Reward: {episode_reward:.2f}, Steps: {steps}")

    return episode_reward, steps


def record_comparison(
    baseline_path: str = "models/baseline/a2c_frozenlake_baseline_final",
    icm_path: str = "models/icm/a2c_frozenlake_icm_final",
    n_episodes: int = 5,
    video_folder: str = "videos/comparison",
    fps: int = 30,
):
    """Record episodes from both baseline and ICM models for side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print("Recording Comparison Videos")
    print(f"{'=' * 60}\n")

    baseline_stats = record_episodes(
        model_path=baseline_path,
        episode_numbers=list(range(1, n_episodes + 1)),
        video_folder=f"{video_folder}/baseline",
        fps=fps,
        model_type='baseline',
    )

    icm_stats = record_episodes(
        model_path=icm_path,
        episode_numbers=list(range(1, n_episodes + 1)),
        video_folder=f"{video_folder}/icm",
        fps=fps,
        model_type='icm',
    )

    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print(f"{'=' * 60}")

    baseline_rewards = [s['reward'] for s in baseline_stats]
    icm_rewards = [s['reward'] for s in icm_stats]

    print("\nBaseline A2C:")
    print(f"  Mean Reward: {np.mean(baseline_rewards):.2f} ± {np.std(baseline_rewards):.2f}")
    print(f"  Mean Steps: {np.mean([s['steps'] for s in baseline_stats]):.1f}")

    print("\nA2C + ICM:")
    print(f"  Mean Reward: {np.mean(icm_rewards):.2f} ± {np.std(icm_rewards):.2f}")
    print(f"  Mean Steps: {np.mean([s['steps'] for s in icm_stats]):.1f}")

    print(f"\n{'=' * 60}\n")

    return {
        'baseline': baseline_stats,
        'icm': icm_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record videos of FrozenLake A2C agents')
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'comparison', 'both'],
        help='Recording mode',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['baseline', 'icm'],
        default='baseline',
        help='Model type to record (for single mode)',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model (without .zip)',
    )
    parser.add_argument(
        '--baseline-path',
        type=str,
        default='models/baseline/a2c_frozenlake_baseline_final',
        help='Path to baseline model',
    )
    parser.add_argument(
        '--icm-path',
        type=str,
        default='models/icm/a2c_frozenlake_icm_final',
        help='Path to ICM model',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        nargs='+',
        help='Episode numbers to record',
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=5,
        help='Number of episodes for comparison mode',
    )
    parser.add_argument(
        '--video-folder',
        type=str,
        default='videos',
        help='Folder to save videos',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for recorded videos',
    )

    args = parser.parse_args()

    if args.mode == 'single':
        model_path = args.model_path
        if model_path is None:
            model_path = args.baseline_path if args.model_type == 'baseline' else args.icm_path

        episodes = args.episodes or [1, 5, 10]
        print(f"Recording {args.model_type.upper()} episodes: {episodes}")
        record_episodes(
            model_path=model_path,
            episode_numbers=episodes,
            video_folder=args.video_folder,
            fps=args.fps,
            model_type=args.model_type,
        )

    elif args.mode == 'comparison':
        record_comparison(
            baseline_path=args.baseline_path,
            icm_path=args.icm_path,
            n_episodes=args.n_episodes,
            video_folder=args.video_folder,
            fps=args.fps,
        )

    elif args.mode == 'both':
        episodes = args.episodes or [1, 5, 10]

        print("\n[1/2] Recording Baseline A2C...")
        record_episodes(
            model_path=args.baseline_path,
            episode_numbers=episodes,
            video_folder=f"{args.video_folder}/baseline",
            fps=args.fps,
            model_type='baseline',
        )

        print("\n[2/2] Recording A2C with ICM...")
        record_episodes(
            model_path=args.icm_path,
            episode_numbers=episodes,
            video_folder=f"{args.video_folder}/icm",
            fps=args.fps,
            model_type='icm',
        )

    print("\n💡 Usage examples:")
    print("   python -m curiosity_a2c.record_videos --mode single --model-type baseline --episodes 1 5 10")
    print("   python -m curiosity_a2c.record_videos --mode single --model-type icm --episodes 1 5 10")
    print("   python -m curiosity_a2c.record_videos --mode comparison --n-episodes 5")
    print("   python -m curiosity_a2c.record_videos --mode both --episodes 1 3 5 7 10")
