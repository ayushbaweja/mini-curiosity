import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch


def make_env():
    """Create and wrap the environment"""
    env = gym.make('MountainCar-v0')
    env = Monitor(env)
    return env


def train_baseline_a2c(
    total_timesteps=100_000,
    n_envs=4,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    vf_coef=0.5,
    save_path="a2c_mountaincar"
):
    """
    Train baseline A2C on MountainCar
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to run for each environment per update
        gamma: Discount factor
        gae_lambda: Factor for GAE
        ent_coef: Entropy coefficient for exploration
        vf_coef: Value function coefficient
        save_path: Path to save the model
    """
    
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Normalize observations and rewards for better training
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # Callbacks for evaluation and checkpointing
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/{save_path}/",
        log_path=f"./logs/{save_path}/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./logs/{save_path}/checkpoints/",
        name_prefix="a2c_model"
    )
    
    # Create A2C model
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=0.5,
        use_rms_prop=True,
        normalize_advantage=True,
        verbose=1,
        tensorboard_log=f"./logs/{save_path}/tensorboard/"
    )
    
    print(f"\n{'='*50}")
    print(f"Training Baseline A2C on MountainCar-v0")
    print(f"{'='*50}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"{'='*50}\n")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model and normalization stats
    model.save(f"{save_path}_final")
    env.save(f"{save_path}_vecnormalize.pkl")
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Model saved to: {save_path}_final.zip")
    print(f"Normalization stats saved to: {save_path}_vecnormalize.pkl")
    print(f"{'='*50}\n")
    
    return model, env


def test_model(model_path="a2c_mountaincar_final", n_episodes=10, render=False):
    """
    Test a trained A2C model
    
    Args:
        model_path: Path to the saved model (without .zip extension)
        n_episodes: Number of episodes to test
        render: Whether to render the environment
    """
    # Load the trained model
    model = A2C.load(model_path)
    
    # Create test environment
    if render:
        env = gym.make('MountainCar-v0', render_mode='human')
    else:
        env = gym.make('MountainCar-v0')
    
    # Load normalization stats if available
    try:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(f"{model_path.replace('_final', '')}_vecnormalize.pkl", env)
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("Warning: Normalization stats not found, continuing without normalization")
        env = DummyVecEnv([lambda: env])
    
    episode_rewards = []
    episode_lengths = []
    
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


if __name__ == "__main__":
    # Train the baseline model
    model, env = train_baseline_a2c(
        total_timesteps=100_000,
        n_envs=4,
        learning_rate=7e-4,
        ent_coef=0.01  # Higher entropy for more exploration
    )
    
    # Test the trained model
    print("\nTesting the trained model...")
    test_model("a2c_mountaincar_final", n_episodes=10, render=False)