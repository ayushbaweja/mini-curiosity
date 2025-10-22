import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os


def record_episodes(
    model_path="a2c_mountaincar_final",
    episode_numbers=[1, 5, 10],
    video_folder="videos",
    fps=30
):
    """
    Record specific episodes as video files
    
    Args:
        model_path: Path to saved model (without .zip)
        episode_numbers: List of episode numbers to record (1-indexed)
        video_folder: Folder to save videos
        fps: Frames per second for video
    """
    
    # Create video directory
    os.makedirs(video_folder, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}.zip...")
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
                name_prefix=f"episode_{episode_num}"
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
    print(f"Video Recording Complete!")
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
    video_folder="videos"
):
    """
    Record a single episode with custom name
    
    Args:
        model_path: Path to saved model
        episode_name: Name for the video file
        video_folder: Folder to save video
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
        name_prefix=episode_name
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
    
    print(f"📹 Recording {episode_name}...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        steps += 1
    
    env.close()
    
    print(f"✅ Video saved: {video_folder}/{episode_name}-episode-0.mp4")
    print(f"   Reward: {episode_reward:.2f}, Steps: {steps}")
    
    return episode_reward, steps


if __name__ == "__main__":
    import sys
    
    # Check if specific episodes were requested
    if len(sys.argv) > 1:
        # Parse episode numbers from command line
        episodes = [int(x) for x in sys.argv[1:]]
        print(f"Recording episodes: {episodes}")
    else:
        # Default episodes
        episodes = [1, 5, 10]
        print(f"Recording default episodes: {episodes}")
    
    # Record videos
    record_episodes(
        model_path="a2c_mountaincar_final",
        episode_numbers=episodes,
        video_folder="videos"
    )
    
    print("\n💡 Tip: You can specify episodes like:")
    print("   python record_videos.py 1 5 10")
    print("   python record_videos.py 3 7 15 20")