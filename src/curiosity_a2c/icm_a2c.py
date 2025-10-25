"""
A2C with Intrinsic Curiosity Module (ICM) for FrozenLake-v1.
"""
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch
from torch.optim import Adam

from .utils import make_env, test_model
from .icm_module import ICMModule, ICMCallback


def train_a2c_with_icm(
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    learning_rate: float = 7e-4,
    n_steps: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 1.0,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    icm_lr: float = 1e-3,
    icm_beta: float = 0.2,
    icm_eta: float = 0.01,
    lambda_weight: float = 0.1,
    save_path: str = "models/icm/a2c_frozenlake_icm",
):
    """Train A2C with ICM on the FrozenLake-v1 environment."""
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Ensure output directories exist
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path("logs") / save_path
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "checkpoints").mkdir(parents=True, exist_ok=True)

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
        tensorboard_log=f"./logs/{save_path}/tensorboard/",
    )

    # Create ICM module
    icm_module = ICMModule(
        env.observation_space,
        env.action_space,
        beta=icm_beta,
        eta=icm_eta,
    ).to(model.device)

    icm_optimizer = Adam(icm_module.parameters(), lr=icm_lr)

    # Setup callbacks
    icm_callback = ICMCallback(
        icm_module,
        icm_optimizer,
        lambda_weight=lambda_weight,
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/{save_path}/",
        log_path=f"./logs/{save_path}/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./logs/{save_path}/checkpoints/",
        name_prefix="a2c_icm_model",
    )

    print(f"\n{'=' * 50}")
    print("Training A2C with ICM on FrozenLake-v1")
    print(f"{'=' * 50}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {n_envs}")
    print(f"A2C Learning rate: {learning_rate}")
    print(f"ICM Learning rate: {icm_lr}")
    print(f"ICM Beta (forward weight): {icm_beta}")
    print(f"ICM Eta (reward scale): {icm_eta}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"{'=' * 50}\n")

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[icm_callback, eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save models
    model.save(f"{save_path}_final")
    torch.save(icm_module.state_dict(), f"{save_path}_icm.pth")
    env.save(f"{save_path}_vecnormalize.pkl")

    print(f"\n{'=' * 50}")
    print("Training completed!")
    print(f"A2C Model saved to: {save_path}_final.zip")
    print(f"ICM Module saved to: {save_path}_icm.pth")
    print(f"{'=' * 50}\n")

    return model, icm_module, env


if __name__ == "__main__":
    model, icm, env = train_a2c_with_icm(
        total_timesteps=200_000,
        n_envs=4,
        learning_rate=7e-4,
        ent_coef=0.01,
        icm_lr=1e-3,
        icm_beta=0.2,
        icm_eta=0.01,
    )

    print("\nTesting the trained model...")
    test_model(
        "models/icm/a2c_frozenlake_icm_final",
        n_episodes=10,
        model_type="icm",
        render=False,
    )
