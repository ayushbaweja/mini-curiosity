import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces

class FrozenLakePixelWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(42, 42)):
        super().__init__(env)
        self.shape = shape
        # Obs space (channels, height, width)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.shape[0], self.shape[1]),
            dtype=np.uint8
        )
    
    def observation(self, observation):
        frame = self.env.render()
        frame = cv2.resize(frame, self.shape, interpolation=cv2.INTER_AREA)
        # transpose (H, W, C) to (C, H, W)
        frame = np.transpose(frame, (2, 0, 1))
        return frame