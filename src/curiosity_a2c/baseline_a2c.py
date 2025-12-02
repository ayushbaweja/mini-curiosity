from pathlib import Path
import sys

if __package__ in (None, ''):
    # Running as a script; ensure project src/ is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from curiosity_a2c.envs import make_frozenlake_env


def make_env(render_mode=None):
    """Create and wrap the FrozenLake environment for training or evaluation."""
    return make_frozenlake_env(
        map_name="4x4",
        is_slippery=True,
        render_mode=render_mode,
    )


def train_baseline_a2c(
    total_timesteps=200_000,
    n_envs=4,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    vf_coef=0.5,
    save_path="models/baseline/a2c_frozenlake_baseline"
):
    """
    Train baseline A2C on FrozenLake
    
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
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_obs=10.)
    
    # Ensure output directories are present
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path('logs') / save_path
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / 'checkpoints').mkdir(parents=True, exist_ok=True)

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
        "CnnPolicy",
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
    print(f"Training Baseline A2C on FrozenLake-v1")
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


def test_model(model_path="models/baseline/a2c_frozenlake_baseline_final", n_episodes=10, render=False):
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
    render_mode = 'human' if render else None
    env = DummyVecEnv([lambda: make_env(render_mode=render_mode)])

    # Load normalization stats if available
    try:
        env = VecNormalize.load(f"{model_path.replace('_final', '')}_vecnormalize.pkl", env)
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("Warning: Normalization stats not found, continuing without normalization")
    
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
    print(f"Success Rate: {sum(1 for r in episode_rewards if r > 0.0) / n_episodes * 100:.1f}%")
    print(f"{'='*50}\n")
    
    return episode_rewards, episode_lengths


if __name__ == "__main__":
    # Train the baseline model
    model, env = train_baseline_a2c(
        total_timesteps=200_000,
        n_envs=4,
        learning_rate=7e-4,
        ent_coef=0.01  # Higher entropy for more exploration
    )
    
    # Test the trained model
    print("\nTesting the trained model...")
    test_model("models/baseline/a2c_frozenlake_baseline_final", n_episodes=10, render=False)