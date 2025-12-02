"""
Intrinsic Curiosity Module (ICM) implementation.
Based on: "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module with forward and inverse dynamics models.
    
    Supports both:
    - Visual observations (Conv2D encoder)
    - Vector observations (MLP encoder)
    """
    
    def __init__(self, observation_space, action_space, feature_dim=288, beta=0.2, eta=0.01):
        """
        Initialize ICM module.
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            feature_dim: Dimension of learned feature representation
            beta: Weight for forward loss vs inverse loss (paper uses 0.2)
            eta: Scaling factor for intrinsic reward (paper uses 0.01)
        """
        super(ICMModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.beta = beta
        self.eta = eta
        
        # Determine if observations are images or vectors
        if len(observation_space.shape) == 3:
            # Image observations (C, H, W)
            self.is_image = True
            n_input_channels = observation_space.shape[0]
        elif len(observation_space.shape) == 1:
            # Vector observations (e.g., MountainCar)
            self.is_image = False
            self.obs_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space shape: {observation_space.shape}")
        
        # Determine action space
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
            self.discrete = True
        else:
            self.action_dim = action_space.shape[0]
            self.discrete = False
        
        # Create appropriate feature encoder
        if self.is_image:
            # Convolutional encoder for images (as in paper)
            self.feature_encoder = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
            )
            
            # Calculate feature dimension after convolutions
            with torch.no_grad():
                sample = torch.zeros(1, *observation_space.shape)
                n_flatten = self.feature_encoder(sample).shape[1]
            
            # Project to desired feature dimension
            self.feature_projection = nn.Linear(n_flatten, feature_dim)
        else:
            # MLP encoder for vector observations
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.feature_projection = nn.Linear(128, feature_dim)
        
        # Inverse model: φ(st), φ(st+1) → at
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        
        # Forward model: φ(st), at → φ(st+1)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def encode(self, obs):
        """
        Encode observation to feature space.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Feature representation φ(obs)
        """
        features = self.feature_encoder(obs)
        features = self.feature_projection(features)
        return features
    
    def forward(self, obs, next_obs, action):
        """
        Compute ICM losses and intrinsic reward.
        
        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
        
        Returns:
            Tuple of (forward_loss, inverse_loss, intrinsic_reward)
        """
        # Encode observations to feature space
        phi_obs = self.encode(obs)
        phi_next_obs = self.encode(next_obs)
        
        # Inverse model loss: predict action from state transition
        phi_concat = torch.cat([phi_obs, phi_next_obs], dim=1)
        pred_action = self.inverse_model(phi_concat)
        
        if self.discrete:
            inverse_loss = F.cross_entropy(pred_action, action.long())
        else:
            inverse_loss = F.mse_loss(pred_action, action)
        
        # Forward model loss: predict next state features
        if self.discrete:
            action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        else:
            action_one_hot = action
        
        phi_action = torch.cat([phi_obs, action_one_hot], dim=1)
        pred_phi_next = self.forward_model(phi_action)
        
        forward_loss = F.mse_loss(pred_phi_next, phi_next_obs.detach())
        
        # Intrinsic reward = prediction error in feature space
        intrinsic_reward = self.eta / 2 * torch.norm(
            pred_phi_next - phi_next_obs.detach(), 
            dim=1, 
            p=2
        ) ** 2
        
        return forward_loss, inverse_loss, intrinsic_reward


class ICMCallback(BaseCallback):
    """
    Callback to train ICM module during A2C rollouts and add intrinsic rewards.
    """
    
    def __init__(self, icm_module, icm_optimizer, k_step=1, lambda_weight=0.1, verbose=0):
        """
        Initialize ICM callback.
        
        Args:
            icm_module: ICMModule instance
            icm_optimizer: Optimizer for ICM
            k_step: Number of steps to look ahead
            lambda_weight: Weight for ICM loss (not currently used)
            verbose: Verbosity level
        """
        super(ICMCallback, self).__init__(verbose)
        self.icm_module = icm_module
        self.icm_optimizer = icm_optimizer
        self.k_step = k_step
        self.lambda_weight = lambda_weight
        self.intrinsic_rewards = []
        self.forward_losses = []
        self.inverse_losses = []
        self.icm_losses = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Train ICM on collected rollout data and add intrinsic rewards."""
        rollout_buffer = self.model.rollout_buffer
        buffer_size = rollout_buffer.observations.shape[0]
        n_envs = rollout_buffer.observations.shape[1]

        if buffer_size <= self.k_step:
            return

        obs_t_list = []
        obs_tk_list = []
        action_t_list = []
        
        # Extract transitions step by step
        for step in range(buffer_size - self.k_step):
            for env in range(n_envs):
                obs_t_list.append(rollout_buffer.observations[step, env])
                obs_tk_list.append(rollout_buffer.observations[step + self.k_step, env])
                action_t_list.append(rollout_buffer.actions[step, env])
        
        # Convert to tensors
        obs_t = torch.FloatTensor(np.array(obs_t_list)).to(self.model.device) / 255.0
        obs_tk = torch.FloatTensor(np.array(obs_tk_list)).to(self.model.device) / 255.0
        actions = torch.FloatTensor(np.array(action_t_list)).to(self.model.device)
        
        # Handle action shape for discrete actions (squeeze if needed)
        if len(actions.shape) > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        
        # Train ICM
        self.icm_optimizer.zero_grad()
        forward_loss, inverse_loss, intrinsic_reward = self.icm_module(obs_t, obs_tk, actions)
        
        # Combined ICM loss (Eq. 7 in paper)
        icm_loss = (1 - self.icm_module.beta) * inverse_loss + self.icm_module.beta * forward_loss
        icm_loss.backward()
        self.icm_optimizer.step()
        
        # Add intrinsic rewards to rollout buffer
        intrinsic_reward_np = intrinsic_reward.detach().cpu().numpy()
        intrinsic_reward_reshaped = intrinsic_reward_np.reshape(buffer_size - self.k_step, n_envs)
        rollout_buffer.rewards[:buffer_size - self.k_step] += intrinsic_reward_reshaped
        
        # Track statistics
        self.intrinsic_rewards.extend(intrinsic_reward_np.tolist())
        self.forward_losses.append(forward_loss.item())
        self.inverse_losses.append(inverse_loss.item())
        self.icm_losses.append(icm_loss.item())
        
        # ============================================================
        # LOG TO TENSORBOARD
        # ============================================================
        if self.logger is not None:
            # Log ICM-specific metrics
            self.logger.record("icm/forward_loss", forward_loss.item())
            self.logger.record("icm/inverse_loss", inverse_loss.item())
            self.logger.record("icm/total_loss", icm_loss.item())
            self.logger.record("icm/mean_intrinsic_reward", intrinsic_reward_np.mean())
            self.logger.record("icm/std_intrinsic_reward", intrinsic_reward_np.std())
            self.logger.record("icm/max_intrinsic_reward", intrinsic_reward_np.max())
            self.logger.record("icm/min_intrinsic_reward", intrinsic_reward_np.min())
            
            # Log cumulative statistics
            if len(self.forward_losses) > 0:
                self.logger.record("icm/avg_forward_loss", np.mean(self.forward_losses[-100:]))
                self.logger.record("icm/avg_inverse_loss", np.mean(self.inverse_losses[-100:]))
            
            # Log reward composition
            extrinsic_rewards = rollout_buffer.rewards[:-self.k_step] - intrinsic_reward_reshaped
            self.logger.record("icm/mean_extrinsic_reward", extrinsic_rewards.mean())
            self.logger.record("icm/intrinsic_to_extrinsic_ratio", 
                             intrinsic_reward_np.mean() / (abs(extrinsic_rewards.mean()) + 1e-8))
        
        if self.verbose > 0:
            print(f"ICM - Forward: {forward_loss.item():.4f}, "
                  f"Inverse: {inverse_loss.item():.4f}, "
                  f"Intrinsic Reward: {intrinsic_reward_np.mean():.4f}")