"""
Shared utilities for A2C training and testing
"""
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import A2C


def make_env():
    """Create and wrap the environment"""
    env = gym.make('MountainCar-v0')
    env = Monitor(env)
    return env


def test_model(model_path, n_episodes=10, model_type='baseline'):
    """
    Test a trained A2C model (works for both baseline and ICM versions)
    
    Args:
        model_path: Path to the saved model (without .zip extension)
        n_episodes: Number of episodes to test
        model_type: 'baseline' or 'icm' for proper identification
    """
    # Load the trained model
    model = A2C.load(model_path)
    
    # Create test environment
    env = gym.make('MountainCar-v0')
    
    # Load normalization stats if available
    try:
        env = DummyVecEnv([lambda: env])
        vec_normalize_path = model_path.replace('_final', '') + '_vecnormalize.pkl'
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("Warning: Normalization stats not found, continuing without normalization")
        env = DummyVecEnv([lambda: env])
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n{'='*50}")
    print(f"Testing {model_type.upper()} A2C Model")
    print(f"{'='*50}\n")
    
    for episode in range(n_episodes):
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
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    print(f"\n{'='*50}")
    print(f"Test Results ({n_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} (+/- {np.std(episode_rewards):.2f})")
    print(f"Average Steps: {np.mean(episode_lengths):.2f} (+/- {np.std(episode_lengths):.2f})")
    print(f"Success Rate: {sum(1 for r in episode_rewards if r > -200) / n_episodes * 100:.1f}%")
    print(f"{'='*50}\n")
    
    return episode_rewards, episode_lengths