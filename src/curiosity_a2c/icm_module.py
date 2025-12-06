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
    Modified to support k-step future prediction.
    """
    
    def __init__(self, observation_space, action_space, feature_dim=288, beta=0.2, eta=0.01, k_step=1):
        """
        Initialize ICM module.
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            feature_dim: Dimension of learned feature representation
            beta: Weight for forward loss vs inverse loss
            eta: Scaling factor for intrinsic reward
            k_step: Number of future steps to predict
        """
        super(ICMModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.beta = beta
        self.eta = eta
        self.k_step = k_step
        
        if len(observation_space.shape) == 3:
            self.is_image = True
            n_input_channels = observation_space.shape[0]
        elif len(observation_space.shape) == 1:
            self.is_image = False
            self.obs_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space shape: {observation_space.shape}")
        
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
            self.discrete = True
        else:
            self.action_dim = action_space.shape[0]
            self.discrete = False
        
        if self.is_image:
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
            with torch.no_grad():
                sample = torch.zeros(1, *observation_space.shape)
                n_flatten = self.feature_encoder(sample).shape[1]
            self.feature_projection = nn.Linear(n_flatten, feature_dim)
        else:
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.feature_projection = nn.Linear(128, feature_dim)
        
        # Inverse model: φ(st), φ(st+1) → at
        # (Standard inverse model always predicts action between t and t+1)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        
        # Forward model: φ(st), at → φ(st+1), ..., φ(st+k)
        # Output dimension is scaled by k_step
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim * self.k_step)
        )
    
    def encode(self, obs):
        features = self.feature_encoder(obs)
        features = self.feature_projection(features)
        return features
    
    def forward(self, obs, next_obs_seq, action):
        """
        Compute ICM losses and intrinsic reward for k-step prediction.
        
        Args:
            obs: Current observation (Batch, Obs_Dim)
            next_obs_seq: Sequence of next k observations (Batch, k, Obs_Dim)
            action: Action taken (Batch, Action_Dim)
        
        Returns:
            Tuple of (forward_loss, inverse_loss, intrinsic_reward)
        """
        batch_size = obs.shape[0]
        
        phi_obs = self.encode(obs)
        
        if self.is_image:
            next_obs_flat = next_obs_seq.view(-1, *next_obs_seq.shape[2:])
        else:
            next_obs_flat = next_obs_seq.view(-1, next_obs_seq.shape[-1])
            
        phi_next_flat = self.encode(next_obs_flat)
        
        phi_next_seq = phi_next_flat.view(batch_size, self.k_step, self.feature_dim)
        
        phi_next_t1 = phi_next_seq[:, 0, :]
        phi_concat = torch.cat([phi_obs, phi_next_t1], dim=1)
        pred_action = self.inverse_model(phi_concat)
        
        if self.discrete:
            inverse_loss = F.cross_entropy(pred_action, action.long())
        else:
            inverse_loss = F.mse_loss(pred_action, action)
        
        if self.discrete:
            action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        else:
            action_one_hot = action
        
        phi_action = torch.cat([phi_obs, action_one_hot], dim=1)
        pred_phi_next_flat = self.forward_model(phi_action)
        pred_phi_next_seq = pred_phi_next_flat.view(batch_size, self.k_step, self.feature_dim)
        forward_loss = F.mse_loss(pred_phi_next_seq, phi_next_seq.detach())
        
        prediction_errors = torch.norm(
            pred_phi_next_seq - phi_next_seq.detach(), 
            dim=2, 
            p=2
        ) ** 2
        
        # Average error across k steps to get a single scalar reward per timestep
        intrinsic_reward = (self.eta / 2) * prediction_errors.mean(dim=1)
        
        return forward_loss, inverse_loss, intrinsic_reward


class ICMCallback(BaseCallback):
    """
    Callback to train ICM module during A2C rollouts and add intrinsic rewards.
    Updated to handle k-step sequences.
    """
    
    def __init__(self, icm_module, icm_optimizer, k_step=1, lambda_weight=0.1, verbose=0):
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
        
        all_obs = rollout_buffer.observations
        actions = rollout_buffer.actions
        episode_starts = rollout_buffer.episode_starts
        
        buffer_size = actions.shape[0]
        n_envs = actions.shape[1]
        
        obs_t_list = []
        next_obs_seq_list = []
        actions_list = []
        valid_indices = [] 
        
        for step in range(buffer_size - self.k_step):
            for env in range(n_envs):
                is_broken = False
                for lookahead in range(1, self.k_step + 1):
                    if episode_starts[step + lookahead, env]:
                        is_broken = True
                        break
                
                if is_broken:
                    continue
                
                # Valid sequence found
                obs_t_list.append(all_obs[step, env])
                
                # Collect sequence: step+1 to step+1+k
                # Shape: (k, ...)
                seq = all_obs[step + 1 : step + 1 + self.k_step, env]
                next_obs_seq_list.append(seq)
                
                actions_list.append(actions[step, env])
                valid_indices.append((step, env))
        
        if not obs_t_list:
            return

        device = self.model.device
        
        # obs_t: (Batch, ...)
        obs_t = torch.tensor(np.array(obs_t_list), dtype=torch.float32).to(device)
        
        # next_obs_seq: (Batch, k, ...)
        next_obs_seq = torch.tensor(np.array(next_obs_seq_list), dtype=torch.float32).to(device)
        
        if self.icm_module.is_image:
             obs_t = obs_t / 255.0
             next_obs_seq = next_obs_seq / 255.0
            
        actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.float32).to(device)
        
        if len(actions_tensor.shape) > 1 and actions_tensor.shape[-1] == 1:
            actions_tensor = actions_tensor.squeeze(-1)
        
        # Train ICM
        self.icm_optimizer.zero_grad()
        forward_loss, inverse_loss, intrinsic_reward = self.icm_module(obs_t, next_obs_seq, actions_tensor)
        
        icm_loss = (1 - self.icm_module.beta) * inverse_loss + self.icm_module.beta * forward_loss
        icm_loss.backward()
        self.icm_optimizer.step()
        
        intrinsic_reward_np = intrinsic_reward.detach().cpu().numpy()
        
        batch_intrinsic_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        for idx, (r, c) in enumerate(valid_indices):
            batch_intrinsic_rewards[r, c] = intrinsic_reward_np[idx]
            
        rollout_buffer.rewards += batch_intrinsic_rewards
        
        # Logging
        if self.logger is not None:
            self.logger.record("icm/forward_loss", forward_loss.item())
            self.logger.record("icm/inverse_loss", inverse_loss.item())
            self.logger.record("icm/k_step", self.k_step)
            self.logger.record("icm/mean_intrinsic_reward", intrinsic_reward_np.mean())