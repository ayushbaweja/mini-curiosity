"""
Video recording utilities for A2C models (baseline and ICM)
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os


def record_episodes(
    model_path="models/baseline/a2c_mountaincar_baseline_final",
    episode_numbers=[1, 5, 10],
    video_folder="videos",
    fps=30,
    model_type='baseline'
):
    """
    Record specific episodes as video files
    
    Args:
        model_path: Path to saved model (without .zip)
        episode_numbers: List of episode numbers to record (1-indexed)
        video_folder: Folder to save videos
        fps: Frames per second for video
        model_type: 'baseline' or 'icm' for labeling
    """
    
    # Create video directory
    os.makedirs(video_folder, exist_ok=True)
    
    # Load model
    print(f"Loading {model_type.upper()} model from {model_path}.zip...")
    model = A2C.load(model_path)
    
    # Track all episodes
    episode_stats = []
    videos_recorded = []
    
    for episode_num in range(1, max(episode_numbers) + 1):
        # Create environment with video recording for selected episodes
        if episode_num in episode_numbers:
            print(f"\n📹 Recording Episode {episode_num}...")
            
            # Create environment with video wrapper
            env = gym.make('MountainCar-v0', render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda x: True,  # Record this episode
                name_prefix=f"{model_type}_episode_{episode_num}"
            )
        else:
            # Just run without recording
            env = gym.make('MountainCar-v0')
        
        # Wrap for normalization
        env = DummyVecEnv([lambda: env])
        
        # Load normalization stats
        try:
            env = VecNormalize.load(
                f"{model_path.replace('_final', '')}_vecnormalize.pkl",
                env
            )
            env.training = False
            env.norm_reward = False
        except FileNotFoundError:
            print("Warning: Normalization stats not found")
        
        # Run episode
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            if done:
                break
        
        episode_stats.append({
            'episode': episode_num,
            'reward': episode_reward,
            'steps': steps
        })
        
        if episode_num in episode_numbers:
            print(f"✅ Episode {episode_num}: Reward = {episode_reward:.2f}, Steps = {steps}")
            videos_recorded.append(episode_num)
        
        env.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Video Recording Complete! ({model_type.upper()})")
    print(f"{'='*60}")
    print(f"Videos saved to: ./{video_folder}/")
    print(f"Episodes recorded: {videos_recorded}")
    print(f"\nAll episode statistics:")
    for stat in episode_stats:
        marker = "📹" if stat['episode'] in episode_numbers else "  "
        print(f"{marker} Episode {stat['episode']}: "
              f"Reward = {stat['reward']:.2f}, Steps = {stat['steps']}")
    print(f"{'='*60}\n")
    
    # List video files
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    print("Video files created:")
    for vf in sorted(video_files):
        print(f"  - {video_folder}/{vf}")
    
    return episode_stats


def record_single_episode(
    model_path="a2c_mountaincar_final",
    episode_name="test",
    video_folder="videos",
    model_type='baseline'
):
    """
    Record a single episode with custom name
    
    Args:
        model_path: Path to saved model
        episode_name: Name for the video file
        video_folder: Folder to save video
        model_type: 'baseline' or 'icm' for labeling
    """
    os.makedirs(video_folder, exist_ok=True)
    
    # Load model
    model = A2C.load(model_path)
    
    # Create environment with video recording
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
        name_prefix=f"{model_type}_{episode_name}"
    )
    
    # Wrap for normalization
    env = DummyVecEnv([lambda: env])
    
    try:
        env = VecNormalize.load(
            f"{model_path.replace('_final', '')}_vecnormalize.pkl",
            env
        )
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        pass
    
    # Run episode
    obs = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    print(f"📹 Recording {model_type.upper()} - {episode_name}...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        steps += 1
    
    env.close()
    
    print(f"✅ Video saved: {video_folder}/{model_type}_{episode_name}-episode-0.mp4")
    print(f"   Reward: {episode_reward:.2f}, Steps: {steps}")
    
    return episode_reward, steps


def record_comparison(
    baseline_path="models/baseline/a2c_mountaincar_baseline_final",
    icm_path="models/icm/a2c_mountaincar_icm_final",
    n_episodes=5,
    video_folder="videos/comparison"
):
    """
    Record episodes from both baseline and ICM models for side-by-side comparison
    
    Args:
        baseline_path: Path to baseline model
        icm_path: Path to ICM model
        n_episodes: Number of episodes to record from each
        video_folder: Folder to save comparison videos
    """
    print(f"\n{'='*60}")
    print("Recording Comparison Videos")
    print(f"{'='*60}\n")
    
    # Record baseline episodes
    print("[1/2] Recording Baseline A2C episodes...")
    baseline_stats = record_episodes(
        model_path=baseline_path,
        episode_numbers=list(range(1, n_episodes + 1)),
        video_folder=f"{video_folder}/baseline",
        model_type='baseline'
    )
    
    # Record ICM episodes
    print("\n[2/2] Recording ICM A2C episodes...")
    icm_stats = record_episodes(
        model_path=icm_path,
        episode_numbers=list(range(1, n_episodes + 1)),
        video_folder=f"{video_folder}/icm",
        model_type='icm'
    )
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    
    baseline_rewards = [s['reward'] for s in baseline_stats]
    icm_rewards = [s['reward'] for s in icm_stats]
    
    print(f"\nBaseline A2C:")
    print(f"  Mean Reward: {np.mean(baseline_rewards):.2f} ± {np.std(baseline_rewards):.2f}")
    print(f"  Mean Steps: {np.mean([s['steps'] for s in baseline_stats]):.1f}")
    
    print(f"\nA2C + ICM:")
    print(f"  Mean Reward: {np.mean(icm_rewards):.2f} ± {np.std(icm_rewards):.2f}")
    print(f"  Mean Steps: {np.mean([s['steps'] for s in icm_stats]):.1f}")
    
    print(f"\n{'='*60}\n")
    
    return {
        'baseline': baseline_stats,
        'icm': icm_stats
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Record videos of A2C agents')
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'comparison', 'both'],
        help='Recording mode'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['baseline', 'icm'],
        default='baseline',
        help='Model type to record (for single mode)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model (without .zip)'
    )
    parser.add_argument(
        '--baseline-path',
        type=str,
        default='a2c_mountaincar_baseline_final',
        help='Path to baseline model'
    )
    parser.add_argument(
        '--icm-path',
        type=str,
        default='a2c_mountaincar_icm_final',
        help='Path to ICM model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        nargs='+',
        help='Episode numbers to record'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=5,
        help='Number of episodes for comparison mode'
    )
    parser.add_argument(
        '--video-folder',
        type=str,
        default='videos',
        help='Folder to save videos'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Record single model
        if args.model_path is None:
            args.model_path = args.baseline_path if args.model_type == 'baseline' else args.icm_path
        
        if args.episodes:
            print(f"Recording {args.model_type.upper()} episodes: {args.episodes}")
            record_episodes(
                model_path=args.model_path,
                episode_numbers=args.episodes,
                video_folder=args.video_folder,
                model_type=args.model_type
            )
        else:
            # Default episodes
            episodes = [1, 5, 10]
            print(f"Recording {args.model_type.upper()} default episodes: {episodes}")
            record_episodes(
                model_path=args.model_path,
                episode_numbers=episodes,
                video_folder=args.video_folder,
                model_type=args.model_type
            )
    
    elif args.mode == 'comparison':
        # Record both models for comparison
        record_comparison(
            baseline_path=args.baseline_path,
            icm_path=args.icm_path,
            n_episodes=args.n_episodes,
            video_folder=args.video_folder
        )
    
    elif args.mode == 'both':
        # Record both models separately
        episodes = args.episodes if args.episodes else [1, 5, 10]
        
        print("\n[1/2] Recording Baseline A2C...")
        record_episodes(
            model_path=args.baseline_path,
            episode_numbers=episodes,
            video_folder=f"{args.video_folder}/baseline",
            model_type='baseline'
        )
        
        print("\n[2/2] Recording ICM A2C...")
        record_episodes(
            model_path=args.icm_path,
            episode_numbers=episodes,
            video_folder=f"{args.video_folder}/icm",
            model_type='icm'
        )
    
    print("\n💡 Usage examples:")
    print("   # Record baseline model")
    print("   python -m curiosity_a2c.record_videos --mode single --model-type baseline --episodes 1 5 10")
    print("\n   # Record ICM model")
    print("   python -m curiosity_a2c.record_videos --mode single --model-type icm --episodes 1 5 10")
    print("\n   # Record both for comparison")
    print("   python -m curiosity_a2c.record_videos --mode comparison --n-episodes 5")
    print("\n   # Record both separately")
    print("   python -m curiosity_a2c.record_videos --mode both --episodes 1 3 5 7 10")